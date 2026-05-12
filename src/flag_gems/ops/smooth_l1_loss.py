import logging

import torch
import triton
import triton.language as tl

import flag_gems

logger = logging.getLogger(__name__)

_SMALL_N_ELEMENTS = 65536
_AUTOTUNE_THRESHOLD = 65536


# ---------------------------------------------------------------------------
# Element-wise forward kernels (reduction='none')
# ---------------------------------------------------------------------------


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _elementwise_kernel_autotuned(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    diff = x - y
    ad = tl.abs(diff)
    beta_f = beta.to(tl.float32)
    half_beta = 0.5 * beta_f

    l2_part = 0.5 * diff * diff / beta_f
    l1_part = ad - half_beta
    loss = tl.where(ad < beta_f, l2_part, l1_part)

    tl.store(out_ptr + offsets, loss.to(x.dtype), mask=mask)


@triton.jit
def _elementwise_kernel_small(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    beta,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    diff = x - y
    ad = tl.abs(diff)
    beta_f = beta.to(tl.float32)
    half_beta = 0.5 * beta_f

    l2_part = 0.5 * diff * diff / beta_f
    l1_part = ad - half_beta
    loss = tl.where(ad < beta_f, l2_part, l1_part)

    tl.store(out_ptr + offsets, loss.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Single-block reduction kernel (small tensors)
# ---------------------------------------------------------------------------


@triton.jit
def _small_reduce_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    beta,
    reduction,
    BLOCK_SIZE: tl.constexpr,
    NUM_LOOPS: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([], dtype=tl.float32)

    for i in range(NUM_LOOPS):
        start = i * BLOCK_SIZE
        off = start + offsets
        mask = off < n_elements

        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        y = tl.load(y_ptr + off, mask=mask, other=0.0)

        diff = x - y
        ad = tl.abs(diff)
        beta_f = beta.to(tl.float32)

        l2_part = 0.5 * diff * diff / beta_f
        l1_part = ad - 0.5 * beta_f
        loss = tl.where(ad < beta_f, l2_part, l1_part)
        loss = tl.where(mask, loss.to(tl.float32), 0.0)
        acc += tl.sum(loss, axis=0)

    if reduction == 1:
        acc = acc / n_elements.to(tl.float32)

    tl.store(out_ptr, acc)


# ---------------------------------------------------------------------------
# Persistent grid-stride reduction kernel (large tensors)
# ---------------------------------------------------------------------------

_TARGET_BLOCKS = 256


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _persistent_reduce_kernel(
    x_ptr,
    y_ptr,
    mid_ptr,
    n_elements,
    beta,
    TILES_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    acc = tl.zeros([], dtype=tl.float32)

    for tile in range(TILES_PER_BLOCK):
        start = (pid * TILES_PER_BLOCK + tile) * BLOCK_SIZE
        off = start + offsets
        mask = off < n_elements

        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        y = tl.load(y_ptr + off, mask=mask, other=0.0)

        diff = x - y
        ad = tl.abs(diff)
        beta_f = beta.to(tl.float32)

        l2_part = 0.5 * diff * diff / beta_f
        l1_part = ad - 0.5 * beta_f
        loss = tl.where(ad < beta_f, l2_part, l1_part)
        loss = tl.where(mask, loss.to(tl.float32), 0.0)
        acc += tl.sum(loss, axis=0)

    tl.store(mid_ptr + pid, acc)


@triton.jit
def _final_reduce_kernel(
    mid_ptr,
    out_ptr,
    mid_size,
    n_elements,
    reduction,
    BLOCK_MID: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_MID)
    mask = offsets < mid_size
    mid_vals = tl.load(mid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(mid_vals, axis=0)

    if reduction == 1:
        total = total / n_elements.to(tl.float32)

    tl.store(out_ptr, total)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_reduction(reduction):
    if isinstance(reduction, str):
        reduction = reduction.lower()
        mapping = {"none": 0, "mean": 1, "sum": 2}
        if reduction not in mapping:
            raise ValueError(f"Invalid reduction: {reduction}")
        return mapping[reduction]
    if reduction not in (0, 1, 2):
        raise ValueError("reduction must be 0 (none), 1 (mean), or 2 (sum)")
    return reduction


def _launch_reduction(xb, yb, n_elements, beta, reduction, out):
    if n_elements <= _SMALL_N_ELEMENTS:
        BLOCK_SIZE = min(triton.next_power_of_2(n_elements), 1024)
        num_loops = triton.cdiv(n_elements, BLOCK_SIZE)
        _small_reduce_kernel[(1, 1, 1)](
            xb, yb, out, n_elements, beta, reduction, BLOCK_SIZE, num_loops
        )
        return

    n_blocks = min(_TARGET_BLOCKS, triton.cdiv(n_elements, 1024))
    block_size_min = max(1024, triton.next_power_of_2(n_elements // n_blocks + 1))
    tiles_per_block = triton.cdiv(n_elements, n_blocks * block_size_min)

    mid = torch.empty((n_blocks,), dtype=torch.float32, device=xb.device)
    grid = (n_blocks,)
    _persistent_reduce_kernel[grid](xb, yb, mid, n_elements, beta, tiles_per_block)

    block_mid = triton.next_power_of_2(n_blocks)
    _final_reduce_kernel[(1, 1, 1)](
        mid, out, n_blocks, n_elements, reduction, block_mid
    )


def _prepare_tensors(x, y):
    if x.device != y.device:
        raise ValueError("input and target must be on the same device")
    if x.device.type != flag_gems.device:
        raise RuntimeError(
            f"smooth_l1_loss requires {flag_gems.device} tensors, "
            f"got {x.device.type}"
        )
    common_shape = torch.broadcast_shapes(x.shape, y.shape)
    xb = x.expand(common_shape).contiguous()
    yb = y.expand(common_shape).contiguous()
    return xb, yb, common_shape


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def smooth_l1_loss(x, y, reduction=1, beta=1.0):
    logger.debug("GEMS SMOOTH_L1_LOSS")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)

    xb, yb, common_shape = _prepare_tensors(x, y)
    n_elements = xb.numel()

    if n_elements == 0:
        if reduction == 0:
            return torch.empty(common_shape, device=xb.device, dtype=xb.dtype)
        elif reduction == 2:
            return torch.zeros((), device=xb.device, dtype=xb.dtype)
        else:
            return torch.full((), float("nan"), device=xb.device, dtype=xb.dtype)

    if reduction == 0:
        out = torch.empty(common_shape, device=xb.device, dtype=xb.dtype)
        if n_elements <= _AUTOTUNE_THRESHOLD:
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _elementwise_kernel_small[grid](xb, yb, out, n_elements, beta, BLOCK_SIZE)
        else:
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            _elementwise_kernel_autotuned[grid](xb, yb, out, n_elements, beta)
        return out
    else:
        out = torch.empty((), dtype=xb.dtype, device=xb.device)
        _launch_reduction(xb, yb, n_elements, beta, reduction, out)
        return out


def smooth_l1_loss_out(x, y, reduction=1, beta=1.0, out=None):
    logger.debug("GEMS SMOOTH_L1_LOSS_OUT")
    reduction = _normalize_reduction(reduction)
    beta = float(beta)

    xb, yb, common_shape = _prepare_tensors(x, y)
    n_elements = xb.numel()

    if out is None:
        if reduction == 0:
            out = torch.empty(common_shape, device=xb.device, dtype=xb.dtype)
        else:
            out = torch.empty((), device=xb.device, dtype=xb.dtype)
    else:
        if out.device != xb.device:
            raise ValueError("out tensor device mismatch")
        if reduction == 0:
            if tuple(out.shape) != tuple(common_shape):
                raise ValueError("out tensor shape mismatch for reduction='none'")
        else:
            if out.numel() != 1:
                raise ValueError("out tensor must be scalar for reduced output")

    if n_elements == 0:
        if reduction == 2:
            out.fill_(0)
        elif reduction == 1:
            out.fill_(float("nan"))
        return out

    if reduction == 0:
        if n_elements <= _AUTOTUNE_THRESHOLD:
            BLOCK_SIZE = 1024
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            _elementwise_kernel_small[grid](xb, yb, out, n_elements, beta, BLOCK_SIZE)
        else:
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            _elementwise_kernel_autotuned[grid](xb, yb, out, n_elements, beta)
    else:
        _launch_reduction(xb, yb, n_elements, beta, reduction, out)

    return out
