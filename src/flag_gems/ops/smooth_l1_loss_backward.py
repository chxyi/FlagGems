import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_AUTOTUNE_THRESHOLD = 65536


# ---------------------------------------------------------------------------
# Backward kernels
# ---------------------------------------------------------------------------

# For mean/sum reductions grad_output is a scalar; the kernel broadcasts it
# and applies the 1/N scaling inline so we never allocate a huge expanded
# tensor in Python.


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
    ],
    key=["n_elements"],
)
@triton.jit
def _backward_kernel_autotuned(
    grad_output_ptr,
    x_ptr,
    y_ptr,
    grad_input_ptr,
    n_elements,
    beta,
    reduction,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    if reduction == 0:
        grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
    else:
        # Load scalar, scale, then broadcast to vector — both branches must
        # produce the same shape for Triton's type inference.
        s = tl.load(grad_output_ptr).to(tl.float32)
        if reduction == 1:
            s = s / n_elements.to(tl.float32)
        grad_out = s + tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    diff = x - y
    beta_f = beta.to(tl.float32)

    l2_grad = diff / beta_f
    l1_grad = tl.where(diff >= 0, 1.0, -1.0)
    use_l2 = tl.abs(diff) < beta_f
    g = tl.where(use_l2, l2_grad, l1_grad)

    grad_in = g * grad_out
    tl.store(grad_input_ptr + offsets, grad_in.to(x.dtype), mask=mask)


@triton.jit
def _backward_kernel_small(
    grad_output_ptr,
    x_ptr,
    y_ptr,
    grad_input_ptr,
    n_elements,
    beta,
    reduction,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    if reduction == 0:
        grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
    else:
        s = tl.load(grad_output_ptr).to(tl.float32)
        if reduction == 1:
            s = s / n_elements.to(tl.float32)
        grad_out = s + tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    diff = x - y
    beta_f = beta.to(tl.float32)

    l2_grad = diff / beta_f
    l1_grad = tl.where(diff >= 0, 1.0, -1.0)
    use_l2 = tl.abs(diff) < beta_f
    g = tl.where(use_l2, l2_grad, l1_grad)

    grad_in = g * grad_out
    tl.store(grad_input_ptr + offsets, grad_in.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def smooth_l1_loss_backward(grad_output, x, y, reduction, beta):
    logger.debug("GEMS SMOOTH_L1_LOSS_BACKWARD")

    if x.numel() == 0:
        return torch.zeros_like(x)

    common_shape = torch.broadcast_shapes(x.shape, y.shape)
    needs_reduce = x.shape != common_shape

    xb = x.expand(common_shape).contiguous() if needs_reduce else x.contiguous()
    yb = y.expand(common_shape).contiguous() if needs_reduce else y.contiguous()

    grad_output = grad_output.contiguous()
    n_elements = xb.numel()

    grad_bc = torch.empty(common_shape, dtype=xb.dtype, device=xb.device)

    if n_elements <= _AUTOTUNE_THRESHOLD:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _backward_kernel_small[grid](
            grad_output, xb, yb, grad_bc, n_elements, beta, reduction, BLOCK_SIZE
        )
    else:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _backward_kernel_autotuned[grid](
            grad_output, xb, yb, grad_bc, n_elements, beta, reduction
        )

    if needs_reduce:
        grad_input = grad_bc
        for d in reversed(range(grad_bc.dim())):
            if d >= x.dim() or (
                d < x.dim() and x.shape[d] == 1 and grad_bc.shape[d] > 1
            ):
                grad_input = grad_input.sum(dim=d, keepdim=(d < x.dim()))
        grad_input = grad_input.view(x.shape)
    else:
        grad_input = grad_bc.view(x.shape)

    return grad_input


def smooth_l1_loss_backward_out(grad_output, x, y, reduction, beta, *, grad_input):
    logger.debug("GEMS SMOOTH_L1_LOSS_BACKWARD_GRAD_INPUT")

    if x.numel() == 0:
        grad_input.zero_()
        return grad_input

    if grad_input.device != x.device:
        raise ValueError("grad_input tensor device mismatch")
    if grad_input.shape != x.shape:
        raise ValueError("grad_input tensor shape mismatch")

    common_shape = torch.broadcast_shapes(x.shape, y.shape)
    needs_reduce = x.shape != common_shape

    xb = x.expand(common_shape).contiguous() if needs_reduce else x.contiguous()
    yb = y.expand(common_shape).contiguous() if needs_reduce else y.contiguous()

    grad_output = grad_output.contiguous()
    n_elements = xb.numel()

    grad_bc = torch.empty(common_shape, dtype=xb.dtype, device=xb.device)

    if n_elements <= _AUTOTUNE_THRESHOLD:
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _backward_kernel_small[grid](
            grad_output, xb, yb, grad_bc, n_elements, beta, reduction, BLOCK_SIZE
        )
    else:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _backward_kernel_autotuned[grid](
            grad_output, xb, yb, grad_bc, n_elements, beta, reduction
        )

    if needs_reduce:
        temp = grad_bc
        for d in reversed(range(grad_bc.dim())):
            if d >= x.dim() or (
                d < x.dim() and x.shape[d] == 1 and grad_bc.shape[d] > 1
            ):
                temp = temp.sum(dim=d, keepdim=(d < x.dim()))
        grad_input.copy_(temp.view(x.shape))
    else:
        grad_input.copy_(grad_bc.view(x.shape))

    return grad_input
