from diffusers.models.transformers.transformer_hunyuan_video import *


class MyHunyuanVideoTransformer3DModel(HunyuanVideoTransformer3DModel):

    def add_vpt(self, vpt_mode='deep-add'):
        self.vpt_mode = vpt_mode
        if 'deep-add' in vpt_mode:
            self.vpt_seq_num = int(vpt_mode.split('-')[-1])
            total_layers = len(self.transformer_blocks) + len(self.single_transformer_blocks)
            n_dim = self.config.num_attention_heads * self.config.attention_head_dim
            self.vpt = nn.Parameter(torch.randn(total_layers, n_dim))
            self.vpt_scale = nn.Parameter(torch.ones(total_layers, 1))

    def load_vpt(self, dic):
        self.vpt.data.copy_(dic['vpt'])
        self.vpt_scale.data.copy_(dic['vpt_scale'])
    
    def vpt_state_dict(self):
        return {'vpt': self.vpt, 'vpt_scale': self.vpt_scale}

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        gc_ratio: float = 1.0
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb = self.time_text_embed(timestep, guidance, pooled_projections)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.zeros(
            batch_size, sequence_length, sequence_length, device=hidden_states.device, dtype=torch.bool
        )  # [B, N, N]

        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

        for i in range(batch_size):
            attention_mask[i, : effective_sequence_length[i], : effective_sequence_length[i]] = True

        if len(self.vpt_mode) > 0:
            vpt = self.vpt * self.vpt_scale

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

            for i, block in enumerate(self.transformer_blocks):
                if 'deep-add' in self.vpt_mode:
                    L = self.vpt_seq_num * post_patch_height * post_patch_width
                    hidden_states[:, :L] = hidden_states[:, :L] + vpt[None, i:i+1]

                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            for i, block in enumerate(self.single_transformer_blocks):
                if 'deep-add' in self.vpt_mode:
                    idx = i + len(self.transformer_blocks)
                    L = self.vpt_seq_num * post_patch_height * post_patch_width
                    hidden_states[:, :L] = hidden_states[:, :L] + vpt[None, idx:idx+1]

                ratio = float(i) / len(self.single_transformer_blocks)
                if ratio > gc_ratio:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                    )
                else:
                    hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        attention_mask,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )

        else:
            for i, block in enumerate(self.transformer_blocks):
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )
                #print(f'double {i}, {hidden_states.max():.3f}')

            for i, block in enumerate(self.single_transformer_blocks):
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )
                #print(f'single {i}, {hidden_states.max():.3f}')

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
