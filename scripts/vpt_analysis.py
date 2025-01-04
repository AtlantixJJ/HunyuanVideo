import matplotlib.pyplot as plt
import torch

torch.set_grad_enabled(False)
data_dir = 'expr/vpt-deep-add-1'
indices = [1] + list(range(50, 251, 50))
all_files = [f'{data_dir}/vpt-{i}.pt' for i in indices]
all_files = [torch.load(f, weights_only=True, map_location='cpu') for f in all_files]

all_vpts = [f['vpt'] * f['vpt_scale'] for f in all_files]

# visualize all norms
plt.figure(figsize=(10, 5))
all_norms = [torch.norm(f) for f in all_vpts]
ax = plt.subplot(1, 2, 1)
ax.plot(all_norms)
ax.set_title('Norm of VPT')
ax.set_xlabel('Iteration')
ax.set_ylabel('Norm')
ax = plt.subplot(1, 2, 2)
all_scales = [torch.mean(f['vpt_scale']) for f in all_files]
ax.plot(all_scales)
ax.set_title('Scale of VPT')
ax.set_xlabel('Iteration')
ax.set_ylabel('Scale')
plt.tight_layout()
plt.savefig('expr/analysis/vpt_all_norm.png')
plt.close()