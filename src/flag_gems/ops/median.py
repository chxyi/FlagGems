import logging
from collections import namedtuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def median_kernel(
    inp,
    out_val,
    out_idx,
    N,
    BLOCK_N: tl.constexpr,
):
    """Radix-select median using int32 masks throughout for efficiency."""
    pid = ext.program_id(0)
    n_offset = tl.arange(0, BLOCK_N)
    mask = n_offset < N

    raw_vals = tl.load(inp + pid * N + n_offset, mask=mask, other=0)
    indices = n_offset

    f32 = raw_vals.to(tl.float32)
    ui = f32.to(tl.uint32, bitcast=True)
    si = f32.to(tl.int32, bitcast=True)
    one_u32 = tl.full([], value=1, dtype=tl.uint32)
    sign_bit = one_u32 << 31
    shift31 = tl.full([], value=31, dtype=tl.int32)
    sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
    conv_mask = sign_bit | sign_extend
    vals_u32 = ui ^ conv_mask

    k = (N - 1) // 2
    # Use int32 masks: 1 = candidate, 0 = eliminated
    candidates = mask.to(tl.int32)

    # Process 20 MSB bits. For N <= 4096, 2^20 = 1M buckets ≫ N.
    # Collisions are rare, and tie-breaking picks the first candidate.
    for bit in range(31, 11, -1):
        bit_val = (vals_u32 >> bit) & 1  # uint32, 0 or 1
        bit_int = bit_val.to(tl.int32)  # int32, 0 or 1
        zeros = candidates & (1 - bit_int)
        count_zeros = tl.sum(zeros, axis=0)
        keep_zeros = count_zeros > k

        candidates = tl.where(keep_zeros, zeros, candidates & bit_int)
        k = tl.where(keep_zeros, k, k - count_zeros)

    # Extract the median: pick the first remaining candidate
    candidate_positions = tl.where(candidates != 0, indices, BLOCK_N + 1)
    best_pos = tl.min(candidate_positions, axis=0)

    median_val = tl.sum(
        tl.where(
            tl.arange(0, BLOCK_N) == best_pos,
            raw_vals,
            tl.zeros([BLOCK_N], dtype=raw_vals.dtype),
        ),
        axis=0,
    )
    median_idx = best_pos.to(tl.int64)

    tl.store(out_val + pid, median_val)
    tl.store(out_idx + pid, median_idx.to(tl.int64))


@libentry()
@triton.jit
def median_kernel_fp16(
    inp,
    out_val,
    out_idx,
    N,
    BLOCK_N: tl.constexpr,
):
    """Radix-select optimized for fp16: only 16 MSB bits cover all fp16 precision."""
    pid = ext.program_id(0)
    n_offset = tl.arange(0, BLOCK_N)
    mask = n_offset < N

    raw_vals = tl.load(inp + pid * N + n_offset, mask=mask, other=0)
    indices = n_offset

    f32 = raw_vals.to(tl.float32)
    ui = f32.to(tl.uint32, bitcast=True)
    si = f32.to(tl.int32, bitcast=True)
    one_u32 = tl.full([], value=1, dtype=tl.uint32)
    sign_bit = one_u32 << 31
    shift31 = tl.full([], value=31, dtype=tl.int32)
    sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
    conv_mask = sign_bit | sign_extend
    vals_u32 = ui ^ conv_mask

    k = (N - 1) // 2
    candidates = mask.to(tl.int32)

    for bit in range(31, 15, -1):
        bit_val = (vals_u32 >> bit) & 1
        bit_int = bit_val.to(tl.int32)
        zeros = candidates & (1 - bit_int)
        count_zeros = tl.sum(zeros, axis=0)
        keep_zeros = count_zeros > k
        candidates = tl.where(keep_zeros, zeros, candidates & bit_int)
        k = tl.where(keep_zeros, k, k - count_zeros)

    candidate_positions = tl.where(candidates != 0, indices, BLOCK_N + 1)
    best_pos = tl.min(candidate_positions, axis=0)

    median_val = tl.sum(
        tl.where(
            tl.arange(0, BLOCK_N) == best_pos,
            raw_vals,
            tl.zeros([BLOCK_N], dtype=raw_vals.dtype),
        ),
        axis=0,
    )
    median_idx = best_pos.to(tl.int64)

    tl.store(out_val + pid, median_val)
    tl.store(out_idx + pid, median_idx.to(tl.int64))


@libentry()
@triton.jit
def median_kernel_small(
    inp,
    out_val,
    out_idx,
    N,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Radix-select for small N with multiple rows per block."""
    pid = ext.program_id(0)
    row_ids = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)[:, None]
    n_offset = tl.arange(0, BLOCK_N)[None, :]
    mask = n_offset < N

    offsets = row_ids * N + n_offset
    raw_vals = tl.load(inp + offsets, mask=mask, other=0)

    f32 = raw_vals.to(tl.float32)
    ui = f32.to(tl.uint32, bitcast=True)
    si = f32.to(tl.int32, bitcast=True)
    one_u32 = tl.full([], value=1, dtype=tl.uint32)
    sign_bit = one_u32 << 31
    shift31 = tl.full([], value=31, dtype=tl.int32)
    sign_extend = (si >> shift31).to(tl.uint32, bitcast=True)
    conv_mask = sign_bit | sign_extend
    vals_u32 = ui ^ conv_mask

    indices = n_offset

    k = tl.full([ROWS_PER_BLOCK], value=(N - 1) // 2, dtype=tl.int32)
    candidates = ((tl.arange(0, ROWS_PER_BLOCK)[:, None] >= 0) & mask).to(tl.int32)

    for bit in range(31, 11, -1):
        bit_val = (vals_u32 >> bit) & 1
        bit_int = bit_val.to(tl.int32)
        zeros = candidates & (1 - bit_int)
        count_zeros = tl.sum(zeros, axis=1)
        keep_zeros = count_zeros > k

        candidates = tl.where(keep_zeros[:, None], zeros, candidates & bit_int)
        k = tl.where(keep_zeros, k, k - count_zeros)

    # Extract the first candidate per row
    candidate_positions = tl.where(candidates != 0, indices, BLOCK_N + 1)
    best_pos = tl.min(candidate_positions, axis=1)

    median_val = tl.sum(
        tl.where(
            tl.arange(0, BLOCK_N)[None, :] == best_pos[:, None],
            raw_vals,
            tl.zeros([ROWS_PER_BLOCK, BLOCK_N], dtype=raw_vals.dtype),
        ),
        axis=1,
    )
    median_idx = best_pos.to(tl.int64)

    out_offsets = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    tl.store(out_val + out_offsets, median_val)
    tl.store(out_idx + out_offsets, median_idx.to(tl.int64))


@libentry()
@triton.jit
def median_kernel_tiled(
    inp,
    out_val,
    out_idx,
    N,
    BLOCK_N: tl.constexpr,
    N_ITERS: tl.constexpr,
):
    """Binary-search median for large N with configurable iterations."""
    pid = ext.program_id(0)

    in_dtype = inp.dtype.element_ty
    max_init = float("inf")
    min_init = float("-inf")

    lo = tl.full([], value=max_init, dtype=tl.float32)
    hi = tl.full([], value=min_init, dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        n_offset = start + tl.arange(0, BLOCK_N)
        tmask = n_offset < N
        block_vals = tl.load(inp + pid * N + n_offset, mask=tmask, other=0)
        f32 = block_vals.to(tl.float32)
        lo = tl.minimum(lo, tl.min(tl.where(tmask, f32, max_init)))
        hi = tl.maximum(hi, tl.max(tl.where(tmask, f32, min_init)))

    k = (N - 1) // 2

    for _ in range(N_ITERS):
        mid = (lo + hi) * 0.5
        count_le = tl.zeros([], dtype=tl.int32)
        for start in range(0, N, BLOCK_N):
            n_offset = start + tl.arange(0, BLOCK_N)
            tmask = n_offset < N
            block_vals = tl.load(inp + pid * N + n_offset, mask=tmask, other=0)
            f32 = block_vals.to(tl.float32)
            count_le += tl.sum(tl.where(tmask, f32 <= mid, False).to(tl.int32), axis=0)
        lo = tl.where(count_le <= k, mid, lo)
        hi = tl.where(count_le <= k, hi, mid)

    # The k-th element is the smallest value strictly > lo.
    # Scan for the minimum f32 among elements > lo.
    best_val = tl.full([], value=max_init, dtype=tl.float32)
    best_elem = tl.full([], value=0, dtype=in_dtype)
    best_idx = tl.full([], value=0, dtype=tl.int64)

    for start in range(0, N, BLOCK_N):
        n_offset = start + tl.arange(0, BLOCK_N)
        tmask = n_offset < N
        block_vals = tl.load(inp + pid * N + n_offset, mask=tmask, other=0)
        f32 = block_vals.to(tl.float32)
        # Mask: valid elements >= lo (handle all-same edge case)
        candidate_mask = tmask & (f32 >= lo)
        candidate_vals = tl.where(candidate_mask, f32, max_init)
        local_best_val = tl.min(candidate_vals, axis=0)
        local_pos = tl.argmin(candidate_vals, axis=0)
        update = local_best_val < best_val
        best_val = tl.where(update, local_best_val, best_val)
        local_elem = tl.sum(
            tl.where(
                tl.arange(0, BLOCK_N) == local_pos,
                block_vals,
                tl.zeros([BLOCK_N], dtype=block_vals.dtype),
            ),
            axis=0,
        ).to(block_vals.dtype)
        best_elem = tl.where(update, local_elem, best_elem)
        best_idx = tl.where(update, start + local_pos, best_idx).to(tl.int64)

    tl.store(out_val + pid, best_elem)
    tl.store(out_idx + pid, best_idx.to(tl.int64))


def _median_impl(inp_2d):
    M, N = inp_2d.shape
    dtype = inp_2d.dtype
    out_val = torch.empty((M,), dtype=dtype, device=inp_2d.device)
    out_idx = torch.empty((M,), dtype=torch.int64, device=inp_2d.device)

    with torch_device_fn.device(inp_2d.device):
        if N <= 32:
            block_n = max(triton.next_power_of_2(N), 32)
            rows_per_block = max(1, min(8, 256 // block_n))
            grid = triton.cdiv(M, rows_per_block)
            median_kernel_small[(grid, 1, 1)](
                inp_2d,
                out_val,
                out_idx,
                N,
                rows_per_block,
                block_n,
            )
        elif N <= 4096:
            block_n = max(triton.next_power_of_2(N), 64)
            if dtype == torch.float16:
                median_kernel_fp16[(M, 1, 1)](inp_2d, out_val, out_idx, N, block_n)
            else:
                median_kernel[(M, 1, 1)](inp_2d, out_val, out_idx, N, block_n)
        else:
            if M == 1:
                # Single row: sort is faster (multiple blocks, better GPU utilization)
                from flag_gems.ops.sort import sort_stable

                sorted_vals, sorted_indices = sort_stable(
                    inp_2d,
                    stable=True,
                    dim=-1,
                    descending=False,
                )
                k = (N - 1) // 2
                out_val = sorted_vals[:, k].contiguous()
                out_idx = sorted_indices[:, k].contiguous()
            else:
                block_n = min(triton.next_power_of_2(N), 4096)
                # Per-dtype optimal: bf16 has 7-bit mantissa, needs fewest iters
                if dtype == torch.bfloat16:
                    n_iters = 10
                elif dtype == torch.float16:
                    n_iters = 13
                else:
                    n_iters = 14
                median_kernel_tiled[(M, 1, 1)](
                    inp_2d,
                    out_val,
                    out_idx,
                    N,
                    block_n,
                    n_iters,
                )

    return out_val, out_idx


def median(inp):
    logger.debug("GEMS MEDIAN")
    N = inp.numel()
    if N == 0:
        raise RuntimeError("median() operation is not supported for empty tensors")
    if N == 1:
        return inp.flatten()[0]

    vals, _ = _median_impl(inp.reshape(1, N))
    return vals.reshape([])


def median_dim(inp, dim, keepdim=False):
    logger.debug("GEMS MEDIAN DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    dim = dim % inp.ndim
    shape = list(inp.shape)
    N = shape[dim]

    if N == 0:
        raise RuntimeError(
            "median() operation is not supported for empty tensors along the "
            "specified dimension"
        )

    inp_compressed = dim_compress(inp, dim)
    M = inp_compressed.numel() // N
    inp_2d = inp_compressed.reshape(M, N)

    out_val_flat, out_idx_flat = _median_impl(inp_2d)

    shape[dim] = 1
    out_val = out_val_flat.reshape(shape)
    out_idx = out_idx_flat.reshape(shape)

    if not keepdim:
        out_val = out_val.squeeze(dim)
        out_idx = out_idx.squeeze(dim)

    Median_out = namedtuple("median", ["values", "indices"])
    return Median_out(values=out_val, indices=out_idx)
