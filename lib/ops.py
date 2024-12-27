"""Pytorch operation functions."""
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import gaussian_blur
import torch
import numpy as np
from einops import rearrange
from jaxtyping import Float
from torch import Tensor



##### Sparse Operations #####


def farthest_point_sampling(points, num_samples,
                            replace=False,
                            dist_mode='mean',
                            sampled_indices=None):
    """
    Performs farthest point sampling on a set of points.
    Low efficiency, do not use when num_samples is large.
    Args:
        points: A NumPy array of shape (num_points, dimension) representing the points.
        num_samples: The number of points to sample.
        sampled_indices: A list of indices that have already been sampled.
        dist_mode: 'mean' or 'min', the distance of new point to existing points.
        replace: Whether to sample with replacement (repeated points are allowed).
    Returns:
        sampled indices
    """

    # Check input validity
    if not replace and num_samples > len(points):
        raise ValueError("Number of samples cannot be greater than the number of points when replace=False!")

    N = points.shape[0]
    indices = np.arange(N)

    reduce_fn = lambda x: np.min(x, axis=-1)
    if dist_mode == 'mean':
        reduce_fn = lambda x: np.mean(x, axis=-1)

    # Randomly choose the first point
    if sampled_indices is None:
        sampled_indices = [np.random.randint(0, N, size=1)[0]]

    # Calculate distances to the sampled points
    diff = points[:, None] - points[sampled_indices][None]
    dist2 = reduce_fn((diff ** 2).sum(-1))
    p = dist2 / dist2.sum().clip(min=1e-6)

    # Iteratively select farthest points
    for _ in range(len(sampled_indices), num_samples):
        sampled_idx = int(np.random.choice(indices, size=1, p=p))
        while not replace and sampled_idx in sampled_indices:
            sampled_idx = int(np.random.choice(indices, size=1, p=p))

        # Add the farthest point to the sampled set
        sampled_indices.append(sampled_idx)

        diff = points[:, None] - points[sampled_indices][None]
        dist2 = reduce_fn((diff ** 2).sum(-1))
        p = dist2 / dist2.sum()

    return sampled_indices


def nearest_upsample_coord(indices, up_dims=[-2, -1], scale_factor=2):
    """
    Upsample a sparse tensor using nearest neighbor interpolation.
    Args:
        indices (torch.Tensor): Input indices, (M, N).
        scale_factor (int): Scale factor
    Returns:
        torch.Tensor: Upsampled indices, (2, ssN)
    """
    # upsampling is restricted to 2D case
    assert len(up_dims) == 2
    S = scale_factor
    offsets = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
    offsets = torch.stack(offsets, dim=0).to(indices) # (2, s, s)

    indices = rearrange(indices, 'm n -> m n () ()')
    up_indices = indices[up_dims].repeat(1, 1, S, S)
    up_indices = up_indices * S + offsets[:, None] # (2, N, S, S)
    up_indices = up_indices.view(len(up_dims), -1)

    # other indices are kept the same
    new_indices = indices.repeat(1, 1, S, S)
    new_indices = new_indices.view(len(indices), -1)
    new_indices[up_dims] = up_indices

    return new_indices


def view_merge_coo(x, *dim_group_list):
    """
    Merging consecutive dimensions of a sparse tensor, dim, dim + 1.
    Args:
        x (torch.SparseTensor, COO): 
        dim_group_list: [[dim1, dim2]]. Must be in ascending order.
    Returns:
        torch.sparse_coo_tensor: Sparse tensor
    """
    indices = x.indices()
    values = x.values()

    indices, new_shape = view_merge_coord(indices, x.shape, *dim_group_list)
    return torch.sparse_coo_tensor(indices, values, new_shape)


def view_merge_coord(indices, dims, *dim_group_list):
    """
    Merging consecutive dimensions of a sparse tensor, dim, dim + 1.
    Args:
        x (torch.SparseTensor, COO): 
        dim_group_list: [[dim_st, length], ...]. Must be in ascending order.
    Returns:
        torch.sparse_coo_tensor: Sparse tensor
    """
    
    dims_mult = np.cumprod(dims[::-1], 0)
    dims_mult = np.concatenate([dims_mult[::-1].copy(), np.ones(1)])
    dims_mult = torch.from_numpy(dims_mult).to(indices)

    new_dims, all_indices = [], []
    prev_dim = 0
    for (dim_st, l) in dim_group_list:
        assert dim_st >= prev_dim, "dim_group_list must be in ascending order."

        if dim_st > prev_dim:
            all_indices.append(indices[prev_dim:dim_st])
            new_dims.extend(dims[prev_dim:dim_st])

        mult = dims_mult[dim_st : dim_st + l + 1] / dims_mult[dim_st + l]
        new_indices = indices[dim_st:dim_st + l] * mult[1:, None]
        new_indices = new_indices.sum(0, keepdim=True).long()
        all_indices.append(new_indices)
        new_dims.append(int(mult[0]))

        prev_dim = dim_st + l

    if prev_dim < len(dims):
        all_indices.append(indices[prev_dim:])
        new_dims.extend(dims[prev_dim:])
    return torch.cat(all_indices, dim=0), new_dims


def view_split_coord(indices, dims, *dim_group_list):
    """
    Splitting a dimension of a sparse tensor into multiple dimensions.
    Args:
        x (torch.SparseTensor, COO): 
        dim_group_list: [(dim_orig, [dim1, dim2, ...]), ...]. Must be in asecending order.
    Returns:
        torch.sparse_coo_tensor: Sparse tensor
    """

    new_dims, all_indices = [], []
    prev_dim = 0
    for (dim_orig, dims) in dim_group_list:
        assert dim_orig >= prev_dim, "dim_group_list must be in ascending order."

        if dim_orig > prev_dim:
            all_indices.append(indices[prev_dim:dim_orig])
            new_dims.extend(dims)

        dims_mult = np.cumprod(dims[::-1], 0)
        dims_mult = np.concatenate([dims_mult[::-1][1:].copy(), np.ones(1)])
        dims_mult = torch.from_numpy(dims_mult).to(indices)
        dims = torch.tensor(dims).to(indices)
        new_indices = indices[dim_orig : dim_orig + 1] // dims_mult[:, None]
        new_indices = new_indices % dims[:, None]
        all_indices.append(new_indices)
        new_dims.extend(dims)

        prev_dim = dim_orig + 1

    if prev_dim < len(dims):
        all_indices.append(indices[prev_dim:])
        new_dims.extend(dims[prev_dim:])

    return torch.cat(all_indices, dim=0), new_dims



##### Elementary Functions #####


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


##### Matrix Operations #####


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6),
        dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]

    q = r

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


def build_covariance_lowerdiag(rotation, scale):
    """convert to covariance representation
    """
    # Create world-space covariance matrices.
    # Note that scale need to be un-activated
    cov = build_covariance(scale, F.normalize(rotation))
    shape = cov.shape[:-2]
    return strip_symmetric(cov.view(-1, 3, 3)).view(*shape, 6)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Taken from PyTorch3D.
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Taken from PyTorch3D.
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)



##### Image Operations #####


def erode_and_dilate(x, erode_size=3, dilate_size=11, scale_factor=2):
    """Erode and dilate a mask.
    Args:
        x: mask in [0, 1] scale, (N, 1, H, W).
        erode_size: size of the erode kernel.
        dilate_size: size of the dilate kernel.
        scale_factor: scale factor of the mask.
    Returns:
        A mask in [0, 1] scale. (N, 1, H, W).
    """
    mask = F.interpolate(x,
                    scale_factor=1/scale_factor, mode='bilinear')
    if erode_size > 0:
        mask = gaussian_blur(mask, erode_size)
        mask = (mask > 0.99).float()
    if dilate_size > 0:
        mask = gaussian_blur(mask, dilate_size)
        mask = (mask > 0.01).float()
    mask = F.interpolate(mask,
                    scale_factor=scale_factor, mode='bilinear')
    return (mask > 0.5).float()


def bu(img, size, align_corners=False):
    """Bilinear interpolation with Pytorch.

    Args:
      img : a list of tensors or a tensor.
    """
    if isinstance(img, list):
        return [
            F.interpolate(i, size=size, mode="bilinear", align_corners=align_corners)
            for i in img
        ]
    ndim = img.ndim
    if ndim == 5:
        n1, n2 = img.shape[:2]
        img = img.view(-1, *img.shape[2:])
    res = F.interpolate(img, size=size, mode="bilinear",
                antialias=True, align_corners=align_corners)
    if ndim == 5:
        res = res.view(n1, n2, *res.shape[1:])
    return res


def norm_img(x):
    """Normalize an arbitrary Tensor to [-1, 1]"""
    return (x - x.min()) / (x.max() - x.min()) * 2 - 1


##### Misc Ops #####


def relative_disparity_to_depth(
    relative_disparity: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert relative disparity, where 0 is near and 1 is far, to depth."""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    return 1 / ((1 - relative_disparity) * (disp_near - disp_far) + disp_far + eps)


def depth_to_relative_disparity(
    depth: Float[Tensor, "*#batch"],
    near: Float[Tensor, "*#batch"],
    far: Float[Tensor, "*#batch"],
    eps: float = 1e-10,
) -> Float[Tensor, " *batch"]:
    """Convert depth to relative disparity, where 0 is near and 1 is far"""
    disp_near = 1 / (near + eps)
    disp_far = 1 / (far + eps)
    disp = 1 / (depth + eps)
    return 1 - (disp - disp_far) / (disp_near - disp_far + eps)


def gather_tensor(tensors, cur_device, n_rank, rank):
    """Gather tensors from multiple devices to the first device.
    Args:
        tensors: A list of CPU tensors to be gathered.
        cur_device: The current device.
        n_rank: The number of devices.
        rank: The rank of the current device.
    """
    res = []
    # gather list of tensors to save memory cost
    if isinstance(tensors, list):
        for tensor in tensors:
            d_tensor = tensor.to(cur_device)
            all_tensor = [d_tensor.clone().detach() for _ in range(n_rank)] \
                if rank == 0 else []
            torch.distributed.gather(d_tensor, all_tensor)
            if rank == 0:
                res.append(torch.cat([c.cpu() for c in all_tensor]))
    else: # gather a single torch.Tensor
        d_tensors = tensors.to(cur_device)
        res = [d_tensors.clone().detach() for _ in range(n_rank)] \
                if rank == 0 else []
        torch.distributed.gather(d_tensors, res)
        if rank == 0:
            res = torch.cat(res)
    return res


##### Tensor Type Conversion #####


def tocpu(x):
    """Convert to CPU Tensor."""
    return x.clone().detach().cpu()


def copy_tensor(x, grad=False):
    """Copy a tensor."""
    return x.clone().detach().requires_grad_(grad)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def torch2numpy(x):
    """Convert a Tensor to a numpy array."""
    if isinstance(x, float):
        return x
    return x.detach().cpu().numpy()


def torch2image(x, data_range="[-1,1]"):
    """Convert torch tensor in [-1, 1] scale to be numpy array format
    image in (N, H, W, C) in [0, 255] scale.
    """
    if data_range == "[-1,1]":
        x = (x.clamp(-1, 1) + 1) / 2
    if data_range == "[0,1]":
        x = x.clamp(0, 1)
    x = (x * 255).cpu().numpy()
    if len(x.shape) == 4:
        x = x.transpose(0, 2, 3, 1)
    elif len(x.shape) == 3:
        x = x.transpose(1, 2, 0)  # (C, H, W)
    return x.astype("uint8")


def image2torch(x):
    """Process [0, 255] (N, H, W, C) numpy array format
    image into [0, 1] scale (N, C, H, W) torch tensor.
    """
    y = torch.from_numpy(x).float() / 255.0
    if len(x.shape) == 3 and x.shape[2] == 3:
        return y.permute(2, 0, 1).unsqueeze(0)
    if len(x.shape) == 4:
        return y.permute(0, 3, 1, 2)
    return 0


def pil2torch(img):
    return image2torch(np.asarray(img))


def torch2pil(img, data_range='[-1,1]'):
    return [Image.fromarray(x) for x in torch2image(img, data_range)]


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)
