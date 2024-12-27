from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.transformers.cogvideox_transformer_3d import Transformer2DModelOutput, CogVideoXTransformer3DModel, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.utils import is_torch_version
from diffusers.loaders import  PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.training_utils import free_memory



class CustomCogVideoXTransformer3DModel(CogVideoXTransformer3DModel):        
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        collect_layers: Optional[Tuple[int]] = None,
        ret_pred: bool = False,
        inject_dict=None,
        gc_ratio=1.0,
    ):
        if collect_layers == []:
            return None, None

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

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        # 3. Transformer blocks
        intermediate_states = {}
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing and float(i) / len(self.transformer_blocks) < gc_ratio:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

            if inject_dict is not None and i in inject_dict:
                dic = inject_dict[i]
                feature, alpha = None, None
                if dic['mode'] in ['affine', 'bottleneck']:
                    hidden_states = hidden_states + dic['feature'].to(hidden_states)
                elif dic['mode'] in ['elementwise', 'layerwise']:
                    alpha = dic['alpha'].to(hidden_states)
                    feature = dic['feature'].to(hidden_states).detach()
                    hidden_states = (1 - alpha) * hidden_states + alpha * feature
                if torch.is_grad_enabled(): # to save memory
                    del inject_dict[i], dic, feature, alpha 

            if collect_layers is not None and i in collect_layers:
                intermediate_states[i] = hidden_states
                if i == collect_layers[-1] and not ret_pred:
                    return intermediate_states

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, intermediate_states)
        return Transformer2DModelOutput(sample=output)


class CogVideoXSimpleControlnet(ModelMixin, ConfigMixin, PeftAdapterMixin):

    @register_to_config
    def __init__(
        self, config
    ):
        super().__init__()
        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_dim = config['attention_head_dim']
        self.inner_dim = config['num_attention_heads'] * config['attention_head_dim']
        self.collect_layers = config['collect_layers']
        self.n_collect = len(self.collect_layers)
        self.num_layers = config['num_layers']
        self.mode = config['mode']
        self.stochastic_layers = config['stochastic_layers']
        self.fixed_condtime = config['fixed_condtime']
        self.bottleneck_dim = config['bottleneck_dim']

        n_layer = self.num_layers if self.stochastic_layers > 0 else self.n_collect
        self.timemed_dim = config['timemed_dim']
        if self.mode == 'affine':
            self.out_projectors = nn.ModuleList(
                [nn.Linear(self.inner_dim + self.timemed_dim, self.inner_dim) for _ in range(n_layer)]
            )
        elif self.mode == 'bottleneck':
            fn = lambda : torch.nn.Sequential(
                nn.Linear(self.inner_dim + self.timemed_dim, self.bottleneck_dim),
                nn.ReLU(),
                nn.Linear(self.bottleneck_dim, self.inner_dim)
            )
            self.out_projectors = nn.ModuleList([fn() for _ in range(n_layer)])
        elif self.mode == 'elementwise':
            init = torch.randn(n_layer, self.inner_dim)
            self.out_projectors = nn.Parameter(init)
        self.weights = nn.Parameter(torch.randn(n_layer))

    def config_dict(self):
        return {
            'num_attention_heads': self.num_attention_heads,
            'attention_head_dim': self.attention_head_dim,
            'mode': self.mode,
            'stochastic_layers': self.stochastic_layers,
            'collect_layers': self.collect_layers,
            'fixed_condtime': self.fixed_condtime,
            'timemed_dim': self.timemed_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'num_layers': self.num_layers,
        }

    def forward(
        self,
        layer_states,
        target_timembed=None,
        cond_timembed=None,
    ):
        inject_dict = {}
        dtype = self.weights.dtype
        device = self.weights.device
        # subtract 4 so that when weights=0 gamma is small
        gamma = torch.sigmoid(self.weights)
        if self.mode in ['affine', 'bottleneck']:
            for hidden_states in layer_states.values():
                break
            L = hidden_states.shape[1]
            #target_timembed_ = target_timembed[None].repeat(1, L, 1)
            #cond_timembed_ = cond_timembed[None].repeat(1, L, 1)
            with torch.no_grad():
                embed = (target_timembed - cond_timembed)[None].repeat(1, L, 1)
            for i, (layer_idx, hidden_states) in enumerate(layer_states.items()):
                idx = layer_idx if self.stochastic_layers > 0 else i
                hidden_states = hidden_states.to(device)
                #x = torch.cat([hidden_states, target_timembed_, cond_timembed_], dim=-1)
                x = torch.cat([hidden_states, embed], dim=-1)
                delta_states = gamma[idx] * self.out_projectors[idx](x)
                inject_dict[layer_idx] = {
                    'feature': delta_states.to(dtype),
                    'mode': self.mode
                }
        elif self.mode == 'elementwise':
            for i, (layer_idx, hidden_states) in enumerate(layer_states.items()):
                idx = layer_idx if self.stochastic_layers > 0 else i
                layer_gamma = torch.sigmoid(self.out_projectors[idx])
                inject_dict[layer_idx] = {
                    'feature': hidden_states,
                    'alpha': gamma[idx] * layer_gamma[None, None],
                    'mode': self.mode
                }
        elif self.mode == 'layerwise':
            for i, (layer_idx, hidden_states) in enumerate(layer_states.items()):
                idx = layer_idx if self.stochastic_layers > 0 else i
                inject_dict[layer_idx] = {
                    'feature': hidden_states,
                    'alpha': gamma[idx],
                    'mode': self.mode
                }
        return inject_dict
