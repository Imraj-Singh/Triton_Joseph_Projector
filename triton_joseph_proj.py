import torch
import triton
import triton.language as tl
import numpy as np
import math
import time
import os

# Check CUDA availability
if not torch.cuda.is_available():
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda")
print(f"Using device: {DEVICE}")

@triton.jit
def ray_cube_intersection_device(orig_x, orig_y, orig_z, rdir_x, rdir_y, rdir_z, bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z):
    """ Calculates ray-AABB intersection returning mask, t_entry, t_exit """
    dtype = orig_x.dtype
    small_epsilon = 1e-9
    smaller_epsilon = 1e-12

    signed_epsilon_x = tl.where(rdir_x >= 0, small_epsilon, -small_epsilon)
    invdir_x = 1.0 / tl.where(tl.abs(rdir_x) < small_epsilon, signed_epsilon_x, rdir_x)
    signed_epsilon_y = tl.where(rdir_y >= 0, small_epsilon, -small_epsilon)
    invdir_y = 1.0 / tl.where(tl.abs(rdir_y) < small_epsilon, signed_epsilon_y, rdir_y)
    signed_epsilon_z = tl.where(rdir_z >= 0, small_epsilon, -small_epsilon)
    invdir_z = 1.0 / tl.where(tl.abs(rdir_z) < small_epsilon, signed_epsilon_z, rdir_z)
    t_lower_x = (bmin_x - orig_x) * invdir_x; t_upper_x = (bmax_x - orig_x) * invdir_x
    t_lower_y = (bmin_y - orig_y) * invdir_y; t_upper_y = (bmax_y - orig_y) * invdir_y
    t_lower_z = (bmin_z - orig_z) * invdir_z; t_upper_z = (bmax_z - orig_z) * invdir_z
    t_min_ax_x = tl.minimum(t_lower_x, t_upper_x); t_max_ax_x = tl.maximum(t_lower_x, t_upper_x)
    t_min_ax_y = tl.minimum(t_lower_y, t_upper_y); t_max_ax_y = tl.maximum(t_lower_y, t_upper_y)
    t_min_ax_z = tl.minimum(t_lower_z, t_upper_z); t_max_ax_z = tl.maximum(t_lower_z, t_upper_z)
    t1 = tl.maximum(tl.maximum(t_min_ax_x, t_min_ax_y), t_min_ax_z)
    t2 = tl.minimum(tl.minimum(t_max_ax_x, t_max_ax_y), t_max_ax_z)
    intersec = (t1 < t2) & (t2 > small_epsilon)
    zero_val = tl.zeros_like(t1)
    t_entry = tl.maximum(zero_val, t1)
    return intersec, t_entry, t2

# --- Autotuning ---
vec_configs = [
    triton.Config({'BLOCK_SIZE_M': 32}, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 32}, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 32}, num_warps=16),
    triton.Config({'BLOCK_SIZE_M': 64}, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 64}, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64}, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 128}, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 128}, num_warps=4), 
    triton.Config({'BLOCK_SIZE_M': 128}, num_warps=8),
]

vec_key = ['num_lors', 'img_nx', 'img_ny', 'img_nz']

@triton.autotune(configs=vec_configs, key=vec_key)
@triton.jit
def joseph3d_fwd_kernel_vec(
    proj_out_ptr, xstart_ptr, xend_ptr, img_ptr, img_origin_ptr, voxsize_ptr,
    num_lors: tl.int32, img_nx: tl.int32, img_ny: tl.int32, img_nz: tl.int32,
    stride_proj_n: tl.int32,
    stride_xs_n: tl.int32, stride_xs_d: tl.int32,
    stride_xe_n: tl.int32, stride_xe_d: tl.int32,
    stride_img_x: tl.int32, stride_img_y: tl.int32, stride_img_z: tl.int32,
    BLOCK_SIZE_M: tl.constexpr
    ):
    """ Block-vectorized Triton kernel for forward projection. """
    pid = tl.program_id(axis=0)

    lor_idx_base = pid * BLOCK_SIZE_M
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    lor_indices = lor_idx_base + offs_m
    lor_mask = lor_indices < num_lors

    xs_offset = lor_indices * stride_xs_n
    xe_offset = lor_indices * stride_xe_n
    xs_x = tl.load(xstart_ptr + xs_offset + 0 * stride_xs_d, mask=lor_mask, other=0.0)
    xs_y = tl.load(xstart_ptr + xs_offset + 1 * stride_xs_d, mask=lor_mask, other=0.0)
    xs_z = tl.load(xstart_ptr + xs_offset + 2 * stride_xs_d, mask=lor_mask, other=0.0)
    xe_x = tl.load(xend_ptr + xe_offset + 0 * stride_xe_d, mask=lor_mask, other=0.0)
    xe_y = tl.load(xend_ptr + xe_offset + 1 * stride_xe_d, mask=lor_mask, other=0.0)
    xe_z = tl.load(xend_ptr + xe_offset + 2 * stride_xe_d, mask=lor_mask, other=0.0)

    orig_x = tl.load(img_origin_ptr + 0)
    orig_y = tl.load(img_origin_ptr + 1)
    orig_z = tl.load(img_origin_ptr + 2)
    vs_x = tl.load(voxsize_ptr + 0)
    vs_y = tl.load(voxsize_ptr + 1)
    vs_z = tl.load(voxsize_ptr + 2)
    dtype = xs_x.dtype
    
    d_x = xe_x - xs_x
    d_y = xe_y - xs_y
    d_z = xe_z - xs_z
    d_sq_x = d_x * d_x
    d_sq_y = d_y * d_y
    d_sq_z = d_z * d_z
    lsq = d_sq_x + d_sq_y + d_sq_z
    bmin_x = -1.0 * vs_x + orig_x
    bmin_y = -1.0 * vs_y + orig_y
    bmin_z = -1.0 * vs_z + orig_z
    bmax_x = img_nx * vs_x + orig_x
    bmax_y = img_ny * vs_y + orig_y
    bmax_z = img_nz * vs_z + orig_z
    intersec, t1, t2 = ray_cube_intersection_device(xs_x, xs_y, xs_z, d_x, d_y, d_z, bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z)

    accum_dtype = dtype
    if dtype == tl.float16:
        accum_dtype = tl.float32
    elif dtype == tl.float32:
        accum_dtype = tl.float64

    accumulated_proj = tl.zeros((BLOCK_SIZE_M,), dtype=accum_dtype)
    active_lor_mask_init = lor_mask & intersec

    safe_lsq = tl.maximum(lsq, 1e-12)
    len_lor = tl.sqrt(safe_lsq)

    safe_d_sq_x = tl.where(lsq < 1e-12, 1.0, d_sq_x)
    safe_d_sq_y = tl.where(lsq < 1e-12, 0.0, d_sq_y)
    safe_d_sq_z = tl.where(lsq < 1e-12, 0.0, d_sq_z)

    is_x_dom = (safe_d_sq_x >= safe_d_sq_y) & (safe_d_sq_x >= safe_d_sq_z)
    is_y_dom = (safe_d_sq_y > safe_d_sq_x) & (safe_d_sq_y >= safe_d_sq_z)
    
    axis_idx = tl.where(is_x_dom, 0, tl.where(is_y_dom, 1, 2))
    vs_axis = tl.where(is_x_dom, vs_x, tl.where(is_y_dom, vs_y, vs_z))
    orig_axis = tl.where(is_x_dom, orig_x, tl.where(is_y_dom, orig_y, orig_z))
    img_dim_axis = tl.where(is_x_dom, img_nx, tl.where(is_y_dom, img_ny, img_nz)).to(tl.int32)
    xs_axis = tl.where(is_x_dom, xs_x, tl.where(is_y_dom, xs_y, xs_z))
    xe_axis = tl.where(is_x_dom, xe_x, tl.where(is_y_dom, xe_y, xe_z))
    d_axis = tl.where(is_x_dom, d_x, tl.where(is_y_dom, d_y, d_z))

    p1_intersect_x = t1 * d_x + xs_x
    p1_intersect_y = t1 * d_y + xs_y
    p1_intersect_z = t1 * d_z + xs_z
    p2_intersect_x = t2 * d_x + xs_x
    p2_intersect_y = t2 * d_y + xs_y
    p2_intersect_z = t2 * d_z + xs_z
    p1_axis = tl.where(is_x_dom, p1_intersect_x, tl.where(is_y_dom, p1_intersect_y, p1_intersect_z))
    p2_axis = tl.where(is_x_dom, p2_intersect_x, tl.where(is_y_dom, p2_intersect_y, p2_intersect_z))

    safe_abs_d_axis = tl.maximum(tl.abs(d_axis), 1e-9)
    corfac = vs_axis * len_lor / safe_abs_d_axis

    istart_f = (p1_axis - orig_axis) / vs_axis
    iend_f = (p2_axis - orig_axis) / vs_axis

    istart_f_ord = tl.minimum(istart_f, iend_f)
    iend_f_ord = tl.maximum(istart_f, iend_f)

    istart = tl.maximum(0, tl.math.floor(istart_f_ord)).to(tl.int32)
    iend = tl.minimum(img_dim_axis, tl.math.ceil(iend_f_ord)).to(tl.int32)

    max_img_dim = tl.maximum(img_nx, tl.maximum(img_ny, img_nz))

    for i_dom in range(0, max_img_dim):
        iter_range_mask = (i_dom >= istart) & (i_dom < iend)
        active_in_iter_mask = active_lor_mask_init & iter_range_mask
        max_val = tl.max(active_in_iter_mask.to(tl.int32), axis=0)
        is_any_active = (max_val == 1)

        if is_any_active:
            plane_coord = orig_axis + (i_dom.to(dtype)) * vs_axis
            
            signed_epsilon = tl.where(d_axis >= 0, 1e-9, -1e-9)
            safe_d_ax = tl.where(tl.abs(d_axis) < 1e-9, signed_epsilon + 1e-12, d_axis)
            t_plane = (plane_coord - xs_axis) / safe_d_ax
            x_pr_x = t_plane * d_x + xs_x
            x_pr_y = t_plane * d_y + xs_y
            x_pr_z = t_plane * d_z + xs_z

            is_x_dom_vec = axis_idx == 0; is_y_dom_vec = axis_idx == 1
            other_axis1_orig = tl.where(is_x_dom_vec, orig_y, tl.where(is_y_dom_vec, orig_x, orig_x))
            other_axis2_orig = tl.where(is_x_dom_vec, orig_z, tl.where(is_y_dom_vec, orig_z, orig_y))
            other_axis1_vs = tl.where(is_x_dom_vec, vs_y, tl.where(is_y_dom_vec, vs_x, vs_x))
            other_axis2_vs = tl.where(is_x_dom_vec, vs_z, tl.where(is_y_dom_vec, vs_z, vs_y))
            other_axis1_nx = tl.where(is_x_dom_vec, img_ny, tl.where(is_y_dom_vec, img_nx, img_nx)).to(tl.int32)
            other_axis2_nx = tl.where(is_x_dom_vec, img_nz, tl.where(is_y_dom_vec, img_nz, img_ny)).to(tl.int32)
            other_axis1_pr = tl.where(is_x_dom_vec, x_pr_y, tl.where(is_y_dom_vec, x_pr_x, x_pr_x))
            other_axis2_pr = tl.where(is_x_dom_vec, x_pr_z, tl.where(is_y_dom_vec, x_pr_z, x_pr_y))

            x_pr_norm1 = (other_axis1_pr - other_axis1_orig) / other_axis1_vs
            x_pr_norm2 = (other_axis2_pr - other_axis2_orig) / other_axis2_vs
            i_floor1 = tl.math.floor(x_pr_norm1).to(tl.int32)
            i_floor2 = tl.math.floor(x_pr_norm2).to(tl.int32)
            i_ceil1 = i_floor1 + 1
            i_ceil2 = i_floor2 + 1
            tmp1 = x_pr_norm1 - i_floor1.to(dtype)
            tmp2 = x_pr_norm2 - i_floor2.to(dtype)

            voxel_sum = tl.zeros((BLOCK_SIZE_M,), dtype=dtype)

            for k in tl.static_range(4):
                if k == 0:
                    idx1, idx2, w = i_floor1, i_floor2, (1 - tmp1) * (1 - tmp2)
                elif k == 1:
                    idx1, idx2, w = i_ceil1, i_floor2, tmp1 * (1 - tmp2)
                elif k == 2:
                    idx1, idx2, w = i_floor1, i_ceil2, (1 - tmp1) * tmp2
                else:
                    idx1, idx2, w = i_ceil1, i_ceil2, tmp1 * tmp2

                valid_neighbour = (idx1 >= 0) & (idx1 < other_axis1_nx) & (idx2 >= 0) & (idx2 < other_axis2_nx)
                current_mask = active_in_iter_mask & valid_neighbour

                img_offset = tl.where(axis_idx == 0, idx2 * stride_img_z + idx1 * stride_img_y + i_dom * stride_img_x,
                            tl.where(axis_idx == 1, idx2 * stride_img_z + i_dom * stride_img_y + idx1 * stride_img_x,
                                                    i_dom * stride_img_z + idx2 * stride_img_y + idx1 * stride_img_x))

                voxel_val = tl.load(img_ptr + img_offset, mask=current_mask, other=0.0)
                voxel_sum += tl.where(current_mask, voxel_val * w, 0.0)

            accumulated_proj += tl.where(active_in_iter_mask, voxel_sum * corfac, 0.0)

    proj_offset = lor_indices * stride_proj_n
    tl.store(proj_out_ptr + proj_offset, accumulated_proj.to(dtype), mask=lor_mask)

@triton.autotune(configs=vec_configs, key=vec_key)
@triton.jit
def joseph3d_bwd_kernel_vec(
    img_out_ptr, xstart_ptr, xend_ptr, p_ptr, img_origin_ptr, voxsize_ptr,
    num_lors: tl.int32, img_nx: tl.int32, img_ny: tl.int32, img_nz: tl.int32,
    stride_xs_n: tl.int32, stride_xs_d: tl.int32,
    stride_xe_n: tl.int32, stride_xe_d: tl.int32,
    stride_p_n: tl.int32,
    stride_img_x: tl.int32, stride_img_y: tl.int32, stride_img_z: tl.int32,
    BLOCK_SIZE_M: tl.constexpr
    ):
    """ Block-vectorized Triton kernel for backward projection. """

    pid = tl.program_id(axis=0)
    lor_idx_base = pid * BLOCK_SIZE_M
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    lor_indices = lor_idx_base + offs_m
    lor_mask = lor_indices < num_lors

    xs_offset = lor_indices * stride_xs_n
    xe_offset = lor_indices * stride_xe_n
    p_offset = lor_indices * stride_p_n
    xs_x = tl.load(xstart_ptr + xs_offset + 0 * stride_xs_d, mask=lor_mask, other=0.0)
    xs_y = tl.load(xstart_ptr + xs_offset + 1 * stride_xs_d, mask=lor_mask, other=0.0)
    xs_z = tl.load(xstart_ptr + xs_offset + 2 * stride_xs_d, mask=lor_mask, other=0.0)
    xe_x = tl.load(xend_ptr + xe_offset + 0 * stride_xe_d, mask=lor_mask, other=0.0)
    xe_y = tl.load(xend_ptr + xe_offset + 1 * stride_xe_d, mask=lor_mask, other=0.0)
    xe_z = tl.load(xend_ptr + xe_offset + 2 * stride_xe_d, mask=lor_mask, other=0.0)
    p_val = tl.load(p_ptr + p_offset, mask=lor_mask, other=0.0)

    orig_x = tl.load(img_origin_ptr + 0)
    orig_y = tl.load(img_origin_ptr + 1)
    orig_z = tl.load(img_origin_ptr + 2)
    vs_x = tl.load(voxsize_ptr + 0)
    vs_y = tl.load(voxsize_ptr + 1)
    vs_z = tl.load(voxsize_ptr + 2)
    dtype = xs_x.dtype

    d_x = xe_x - xs_x
    d_y = xe_y - xs_y
    d_z = xe_z - xs_z
    d_sq_x = d_x * d_x
    d_sq_y = d_y * d_y
    d_sq_z = d_z * d_z
    lsq = d_sq_x + d_sq_y + d_sq_z
    bmin_x = -1.0 * vs_x + orig_x
    bmin_y = -1.0 * vs_y + orig_y
    bmin_z = -1.0 * vs_z + orig_z
    bmax_x = img_nx * vs_x + orig_x
    bmax_y = img_ny * vs_y + orig_y
    bmax_z = img_nz * vs_z + orig_z
    intersec, t1, t2 = ray_cube_intersection_device(xs_x, xs_y, xs_z, d_x, d_y, d_z, bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z)

    accum_dtype = dtype
    if dtype == tl.float16:
        accum_dtype = tl.float32

    accumulated_proj = tl.zeros((BLOCK_SIZE_M,), dtype=accum_dtype)
    
    active_lor_mask_init = lor_mask & intersec

    safe_lsq = tl.maximum(lsq, 1e-12)
    len_lor = tl.sqrt(safe_lsq)

    safe_d_sq_x = tl.where(lsq < 1e-12, 1.0, d_sq_x)
    safe_d_sq_y = tl.where(lsq < 1e-12, 0.0, d_sq_y)
    safe_d_sq_z = tl.where(lsq < 1e-12, 0.0, d_sq_z)

    is_x_dom = (safe_d_sq_x >= safe_d_sq_y) & (safe_d_sq_x >= safe_d_sq_z)
    is_y_dom = (safe_d_sq_y > safe_d_sq_x) & (safe_d_sq_y >= safe_d_sq_z)
    
    axis_idx = tl.where(is_x_dom, 0, tl.where(is_y_dom, 1, 2))
    vs_axis = tl.where(is_x_dom, vs_x, tl.where(is_y_dom, vs_y, vs_z))
    orig_axis = tl.where(is_x_dom, orig_x, tl.where(is_y_dom, orig_y, orig_z))
    img_dim_axis = tl.where(is_x_dom, img_nx, tl.where(is_y_dom, img_ny, img_nz)).to(tl.int32)
    xs_axis = tl.where(is_x_dom, xs_x, tl.where(is_y_dom, xs_y, xs_z))
    xe_axis = tl.where(is_x_dom, xe_x, tl.where(is_y_dom, xe_y, xe_z))
    d_axis = tl.where(is_x_dom, d_x, tl.where(is_y_dom, d_y, d_z))

    p1_intersect_x = t1 * d_x + xs_x
    p1_intersect_y = t1 * d_y + xs_y
    p1_intersect_z = t1 * d_z + xs_z
    p2_intersect_x = t2 * d_x + xs_x
    p2_intersect_y = t2 * d_y + xs_y
    p2_intersect_z = t2 * d_z + xs_z
    p1_axis = tl.where(is_x_dom, p1_intersect_x, tl.where(is_y_dom, p1_intersect_y, p1_intersect_z))
    p2_axis = tl.where(is_x_dom, p2_intersect_x, tl.where(is_y_dom, p2_intersect_y, p2_intersect_z))

    safe_abs_d_axis = tl.maximum(tl.abs(d_axis), 1e-9)
    corfac = vs_axis * len_lor / safe_abs_d_axis

    istart_f = (p1_axis - orig_axis) / vs_axis
    iend_f = (p2_axis - orig_axis) / vs_axis

    istart_f_ord = tl.minimum(istart_f, iend_f)
    iend_f_ord = tl.maximum(istart_f, iend_f)

    istart = tl.maximum(0, tl.math.floor(istart_f_ord)).to(tl.int32)
    iend = tl.minimum(img_dim_axis, tl.math.ceil(iend_f_ord)).to(tl.int32)

    is_x_dom_vec = (axis_idx == 0)
    is_y_dom_vec = (axis_idx == 1)
    other_axis1_orig = tl.where(is_x_dom_vec, orig_y, tl.where(is_y_dom_vec, orig_x, orig_x))
    other_axis2_orig = tl.where(is_x_dom_vec, orig_z, tl.where(is_y_dom_vec, orig_z, orig_y))
    other_axis1_vs = tl.where(is_x_dom_vec, vs_y, tl.where(is_y_dom_vec, vs_x, vs_x))
    other_axis2_vs = tl.where(is_x_dom_vec, vs_z, tl.where(is_y_dom_vec, vs_z, vs_y))
    other_axis1_nx = tl.where(is_x_dom_vec, img_ny, tl.where(is_y_dom_vec, img_nx, img_nx)).to(tl.int32)
    other_axis2_nx = tl.where(is_x_dom_vec, img_nz, tl.where(is_y_dom_vec, img_nz, img_ny)).to(tl.int32)

    
    signed_epsilon = tl.where(d_axis >= 0, 1e-9, -1e-9)
    safe_d_ax = tl.where(tl.abs(d_axis) < 1e-9, signed_epsilon, d_axis)

    max_img_dim = tl.maximum(img_nx, tl.maximum(img_ny, img_nz))
    
    for i_dom in range(0, max_img_dim):
        iter_range_mask = (i_dom >= istart) & (i_dom < iend)
        active_in_iter_mask = active_lor_mask_init & iter_range_mask
        max_val = tl.max(active_in_iter_mask.to(tl.int32), axis=0)
        is_any_active = (max_val == 1)

        if is_any_active:
            plane_coord = orig_axis + i_dom.to(dtype) * vs_axis

            t_plane = (plane_coord - xs_axis) / safe_d_ax 
            x_pr_x = t_plane * d_x + xs_x
            x_pr_y = t_plane * d_y + xs_y
            x_pr_z = t_plane * d_z + xs_z

            other_axis1_pr = tl.where(is_x_dom_vec, x_pr_y, tl.where(is_y_dom_vec, x_pr_x, x_pr_x))
            other_axis2_pr = tl.where(is_x_dom_vec, x_pr_z, tl.where(is_y_dom_vec, x_pr_z, x_pr_y))

            x_pr_norm1 = (other_axis1_pr - other_axis1_orig) / other_axis1_vs
            x_pr_norm2 = (other_axis2_pr - other_axis2_orig) / other_axis2_vs
            i_floor1 = tl.math.floor(x_pr_norm1).to(tl.int32)
            i_floor2 = tl.math.floor(x_pr_norm2).to(tl.int32)
            i_ceil1 = i_floor1 + 1
            i_ceil2 = i_floor2 + 1
            tmp1 = x_pr_norm1 - i_floor1.to(dtype)
            tmp2 = x_pr_norm2 - i_floor2.to(dtype)

            update_base = p_val * corfac

            for k in tl.static_range(4):
                if k == 0:
                    idx1, idx2, w = i_floor1, i_floor2, (1 - tmp1) * (1 - tmp2)
                elif k == 1:
                    idx1, idx2, w = i_ceil1, i_floor2, tmp1 * (1 - tmp2)
                elif k == 2:
                    idx1, idx2, w = i_floor1, i_ceil2, (1 - tmp1) * tmp2
                else:
                    idx1, idx2, w = i_ceil1, i_ceil2, tmp1 * tmp2
                
                update_val = update_base * w

                valid_neighbour = (idx1 >= 0) & (idx1 < other_axis1_nx) & (idx2 >= 0) & (idx2 < other_axis2_nx)
                current_mask = active_in_iter_mask & valid_neighbour

                img_offset = tl.where(axis_idx == 0, idx2 * stride_img_z + idx1 * stride_img_y + i_dom * stride_img_x,
                            tl.where(axis_idx == 1, idx2 * stride_img_z + i_dom * stride_img_y + idx1 * stride_img_x,
                                                    i_dom * stride_img_z + idx2 * stride_img_y + idx1 * stride_img_x))

                tl.atomic_add(img_out_ptr + img_offset, update_val.to(dtype), mask=current_mask)

def joseph3d_fwd_vec(
    xstart, xend, img,
    img_origin, voxsize, img_dim_tup
    ):
    assert img.is_contiguous()
    dtype = img.dtype; assert xstart.dtype == dtype and xend.dtype == dtype and img_origin.dtype == dtype and voxsize.dtype == dtype
    #assert xstart.device == DEVICE and xend.device == DEVICE and img.device == DEVICE and img_origin.device == DEVICE and voxsize.device == DEVICE

    nlors = xstart.shape[0]
    img_nx, img_ny, img_nz = img_dim_tup
    projection_data = torch.zeros(nlors, dtype=dtype, device=DEVICE)

    def grid(meta):
        return (triton.cdiv(nlors, meta['BLOCK_SIZE_M']),)
    
    joseph3d_fwd_kernel_vec[grid](
        projection_data,
        xstart, xend,
        img, img_origin, voxsize,
        nlors,
        img_nx, img_ny, img_nz,
        projection_data.stride(0),
        xstart.stride(0), xstart.stride(1),
        xend.stride(0), xend.stride(1),
        img.stride(0), img.stride(1), img.stride(2)
    )

    torch.cuda.synchronize(DEVICE)
    return projection_data

def joseph3d_back_vec(
    xstart, xend, proj,
    img_origin, voxsize, img_dim_tup,
    ):

    dtype = proj.dtype; assert xstart.dtype == dtype and xend.dtype == dtype and img_origin.dtype == dtype and voxsize.dtype == dtype
    #assert xstart.device == DEVICE and xend.device == DEVICE and proj.device == DEVICE and img_origin.device == DEVICE and voxsize.device == DEVICE

    nlors = xstart.shape[0]
    img_nx, img_ny, img_nz = img_dim_tup
    img_out = torch.zeros(img_dim_tup, dtype=dtype, device=DEVICE).contiguous()
    assert img_out.is_contiguous()

    def grid(meta):
            return (triton.cdiv(nlors, meta['BLOCK_SIZE_M']),)

    joseph3d_bwd_kernel_vec[grid](
        img_out,
        xstart, xend, proj,
        img_origin, voxsize,
        nlors,
        img_nx, img_ny, img_nz,
        xstart.stride(0), xstart.stride(1),
        xend.stride(0), xend.stride(1),
        proj.stride(0),
        img_out.stride(0), img_out.stride(1), img_out.stride(2)
    )

    torch.cuda.synchronize(DEVICE)
    return img_out


if __name__ == "__main__":
    import numpy as np
    import time
    import array_api_compat.torch as xp

    dict_data = np.load("small_regular_polygon_geom.npy", allow_pickle=True).item()

    # --- Float16 Test ---
    print("\n--- Testing with float16 ---")
    xstart_f16 = xp.asarray(dict_data['xstart']).cuda().half()
    xend_f16 = xp.asarray(dict_data['xend']).cuda().half()
    img_origin_f16 = xp.asarray(dict_data['img_origin']).cuda().half()
    voxel_size_f16 = xp.asarray(dict_data['voxel_size']).cuda().half()
    img_shape = dict_data['img_shape']
    x_gt_f16 = xp.asarray(dict_data['x']).cuda().half()
    meas_f16 = xp.asarray(dict_data['meas']).cuda().half()

    fwd16 = lambda x: joseph3d_fwd_vec(xstart_f16.reshape(-1,3), xend_f16.reshape(-1,3), x, img_origin_f16, voxel_size_f16, img_shape).reshape(meas_f16.shape)
    back16 = lambda y: joseph3d_back_vec(xstart_f16.reshape(-1,3), xend_f16.reshape(-1,3), y.ravel(), img_origin_f16, voxel_size_f16, img_shape)

    print("Triton Forward (float16) - First call (expect compilation/tuning):")
    start = time.time()
    tmp_fwd_f16 = fwd16(x_gt_f16)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s")

    print("Triton Forward (float16) - Second call (expect cache hit):")
    start = time.time()
    for i in range(100):
        tmp_fwd_f16 = fwd16(x_gt_f16)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s")

    print("Triton Backward (float16) - First call (expect compilation/tuning):")
    start = time.time()
    tmp_bwd_f16 = back16(meas_f16)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s")

    print("Triton Backward (float16) - Second call (expect cache hit):")
    start = time.time()
    for i in range(100):
        tmp_bwd_f16 = back16(meas_f16)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s")

    # --- Float32 Test ---
    print("\n--- Testing with float32 ---")
    xstart_f32 = xp.asarray(dict_data['xstart']).cuda().float()
    xend_f32 = xp.asarray(dict_data['xend']).cuda().float()
    img_origin_f32 = xp.asarray(dict_data['img_origin']).cuda().float()
    voxel_size_f32 = xp.asarray(dict_data['voxel_size']).cuda().float()
    img_shape = dict_data['img_shape']
    x_gt_f32 = xp.asarray(dict_data['x']).cuda().float()
    meas_f32 = xp.asarray(dict_data['meas']).cuda().float()

    fwd32 = lambda x: joseph3d_fwd_vec(xstart_f32.reshape(-1,3), xend_f32.reshape(-1,3), x, img_origin_f32, voxel_size_f32, img_shape).reshape(meas_f32.shape)
    back32 = lambda y: joseph3d_back_vec(xstart_f32.reshape(-1,3), xend_f32.reshape(-1,3), y.ravel(), img_origin_f32, voxel_size_f32, img_shape)

    print("Triton Forward (float32) - First call (expect compilation/tuning):")
    start = time.time()
    tmp_fwd_f32 = fwd32(x_gt_f32)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s")

    print("Triton Forward (float32) - Second call (expect cache hit):")
    start = time.time()
    for i in range(100):
        tmp_fwd_f32 = fwd32(x_gt_f32)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s")

    print("Triton Backward (float32) - First call (expect compilation/tuning):")
    start = time.time()
    tmp_bwd_f32 = back32(meas_f32)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s")

    print("Triton Backward (float32) - Second call (expect cache hit):")
    start = time.time()
    for i in range(100):
        tmp_bwd_f32 = back32(meas_f32)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s")

    # --- Float64 Test ---
    print("\n--- Testing with float64 ---")
    xstart_f64 = xp.asarray(dict_data['xstart']).cuda().double()
    xend_f64 = xp.asarray(dict_data['xend']).cuda().double()
    img_origin_f64 = xp.asarray(dict_data['img_origin']).cuda().double()
    voxel_size_f64 = xp.asarray(dict_data['voxel_size']).cuda().double()
    # img_shape is the same
    x_gt_f64 = xp.asarray(dict_data['x']).cuda().double()
    meas_f64 = xp.asarray(dict_data['meas']).cuda().double()

    fwd64 = lambda x: joseph3d_fwd_vec(xstart_f64.reshape(-1,3), xend_f64.reshape(-1,3), x, img_origin_f64, voxel_size_f64, img_shape).reshape(meas_f64.shape)
    back64 = lambda y: joseph3d_back_vec(xstart_f64.reshape(-1,3), xend_f64.reshape(-1,3), y.ravel(), img_origin_f64, voxel_size_f64, img_shape)

    print("Triton Forward (float64) - First call (expect compilation/tuning):")
    start = time.time()
    tmp_fwd_f64 = fwd64(x_gt_f64)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s") # Should be slow again

    print("Triton Forward (float64) - Second call (expect cache hit):")
    start = time.time()
    for i in range(100):
        tmp_fwd_f64 = fwd64(x_gt_f64)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s") # Should be fast

    print("Triton Backward (float64) - First call (expect compilation/tuning):")
    start = time.time()
    tmp_bwd_f64 = back64(meas_f64)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s") # Should be slow again

    print("Triton Backward (float64) - Second call (expect cache hit):")
    start = time.time()
    for i in range(100):
        tmp_bwd_f64 = back64(meas_f64)
    torch.cuda.synchronize(DEVICE)
    end_t = time.time()
    print(f"Time: {(end_t - start):.6f} s") # Should be fast