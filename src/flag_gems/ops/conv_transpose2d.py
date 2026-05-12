import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.conv2d import conv2d as flag_gems_conv2d
from flag_gems.ops.conv2d import conv2d_backward_kernel_weight, conv2d_forward_kernel
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def conv_transpose2d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
) -> int:
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )


# Phase-separated gather kernel.
# For each (ph, pw) phase, the valid (kh, kw) pairs are determined at compile time,
# eliminating 75% of wasted computation for stride=2 without any atomic operations.
@libentry()
@triton.autotune(
    configs=[
        # CI=32, CO=16 — tiny channels
        triton.Config(
            {"BLOCK_NHW": 64, "BLOCK_CI": 32, "BLOCK_CO": 16}, num_warps=2, num_stages=3
        ),
        # CI=32, CO=32 — balanced
        triton.Config(
            {"BLOCK_NHW": 64, "BLOCK_CI": 32, "BLOCK_CO": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_NHW": 128, "BLOCK_CI": 32, "BLOCK_CO": 32},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_NHW": 256, "BLOCK_CI": 32, "BLOCK_CO": 32},
            num_warps=8,
            num_stages=4,
        ),
        # CI=64, CO=32 — C_in≈64 shapes
        triton.Config(
            {"BLOCK_NHW": 64, "BLOCK_CI": 64, "BLOCK_CO": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_NHW": 128, "BLOCK_CI": 64, "BLOCK_CO": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_NHW": 256, "BLOCK_CI": 64, "BLOCK_CO": 32},
            num_warps=8,
            num_stages=4,
        ),
        # CI=32, CO=64 — large C_out
        triton.Config(
            {"BLOCK_NHW": 64, "BLOCK_CI": 32, "BLOCK_CO": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_NHW": 128, "BLOCK_CI": 32, "BLOCK_CO": 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_NHW": 256, "BLOCK_CI": 32, "BLOCK_CO": 64},
            num_warps=8,
            num_stages=4,
        ),
        # CI=64, CO=64 — balanced large
        triton.Config(
            {"BLOCK_NHW": 64, "BLOCK_CI": 64, "BLOCK_CO": 64}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_NHW": 128, "BLOCK_CI": 64, "BLOCK_CO": 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_NHW": 256, "BLOCK_CI": 64, "BLOCK_CO": 64},
            num_warps=8,
            num_stages=4,
        ),
        # CI=64, CO=128 — huge C_out
        triton.Config(
            {"BLOCK_NHW": 128, "BLOCK_CI": 64, "BLOCK_CO": 128},
            num_warps=8,
            num_stages=4,
        ),
        # CI=128 — large C_in, use smaller BLOCK_NHW for more blocks / occupancy
        triton.Config(
            {"BLOCK_NHW": 32, "BLOCK_CI": 128, "BLOCK_CO": 32},
            num_warps=4,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_NHW": 32, "BLOCK_CI": 128, "BLOCK_CO": 64},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_NHW": 64, "BLOCK_CI": 128, "BLOCK_CO": 32},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_NHW": 64, "BLOCK_CI": 128, "BLOCK_CO": 64},
            num_warps=8,
            num_stages=4,
        ),
        # CI=32 with small NHW — for small outputs needing more blocks
        triton.Config(
            {"BLOCK_NHW": 32, "BLOCK_CI": 32, "BLOCK_CO": 32}, num_warps=2, num_stages=3
        ),
        triton.Config(
            {"BLOCK_NHW": 32, "BLOCK_CI": 64, "BLOCK_CO": 32}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_NHW": 32, "BLOCK_CI": 32, "BLOCK_CO": 64}, num_warps=4, num_stages=3
        ),
    ],
    key=[
        "in_n",
        "in_per_group_c",
        "out_per_group_c",
        "in_h",
        "in_w",
        "out_h_per_phase",
        "out_w_per_phase",
        "kH",
        "kW",
        "stride_h",
        "stride_w",
        "padding_h",
        "padding_w",
        "dilation_h",
        "dilation_w",
        "groups",
    ],
)
@triton.jit
def conv_transpose2d_forward_kernel(
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    in_n,
    in_h,
    in_w,
    out_h,
    out_w,
    out_h_per_phase,
    out_w_per_phase,
    input_n_stride,
    input_c_stride,
    input_height_stride,
    input_width_stride,
    weight_cin_stride,
    weight_cout_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_height_stride,
    output_width_stride,
    in_per_group_c: tl.constexpr,
    out_per_group_c: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    ph: tl.constexpr,
    pw: tl.constexpr,
    groups: tl.constexpr,
    BLOCK_NHW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_g = tl.program_id(2)

    # Decode output positions within this phase's output sub-grid
    nhw = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    n = nhw // (out_h_per_phase * out_w_per_phase)
    hw_local = nhw % (out_h_per_phase * out_w_per_phase)
    h_idx = hw_local // out_w_per_phase
    w_idx = hw_local % out_w_per_phase
    h_out = ph + h_idx * stride_h
    w_out = pw + w_idx * stride_w

    out_c_off = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)

    input_pointer += input_c_stride * pid_g * in_per_group_c
    weight_pointer += weight_cin_stride * pid_g * in_per_group_c

    accum = tl.zeros((BLOCK_NHW, BLOCK_CO), dtype=tl.float32)
    BLOCK_CI_COUNT = (in_per_group_c + BLOCK_CI - 1) // BLOCK_CI

    for kh in tl.static_range(kH):
        for kw in tl.static_range(kW):
            h_mod = (ph + padding_h - kh * dilation_h) % stride_h
            w_mod = (pw + padding_w - kw * dilation_w) % stride_w
            if h_mod == 0 and w_mod == 0:
                h_in = (h_out + padding_h - kh * dilation_h) // stride_h
                w_in = (w_out + padding_w - kw * dilation_w) // stride_w

                valid = (
                    (n < in_n)
                    & (h_in >= 0)
                    & (h_in < in_h)
                    & (w_in >= 0)
                    & (w_in < in_w)
                )

                for ci_blk in range(BLOCK_CI_COUNT):
                    ci = ci_blk * BLOCK_CI + tl.arange(0, BLOCK_CI)
                    valid_ci = ci < in_per_group_c

                    inp = tl.load(
                        input_pointer
                        + n[:, None] * input_n_stride
                        + ci[None, :] * input_c_stride
                        + h_in[:, None] * input_height_stride
                        + w_in[:, None] * input_width_stride,
                        mask=valid[:, None] & valid_ci[None, :],
                        other=0.0,
                    )
                    wt = tl.load(
                        weight_pointer
                        + ci[:, None] * weight_cin_stride
                        + out_c_off[None, :] * weight_cout_stride
                        + kh * weight_height_stride
                        + kw * weight_width_stride,
                        mask=valid_ci[:, None] & (out_c_off < out_per_group_c)[None, :],
                        other=0.0,
                    )
                    accum += tl.dot(inp, wt, allow_tf32=False)

    bias_pointer += pid_g * out_per_group_c + out_c_off
    bias = tl.load(bias_pointer, mask=out_c_off < out_per_group_c, other=0.0).to(
        tl.float32
    )
    accum += bias[None, :]

    out_co = pid_g * out_per_group_c + out_c_off
    tl.store(
        output_pointer
        + n[:, None] * output_n_stride
        + out_co[None, :] * output_c_stride
        + h_out[:, None] * output_height_stride
        + w_out[:, None] * output_width_stride,
        accum.to(output_pointer.dtype.element_ty),
        mask=(n < in_n)[:, None]
        & (out_c_off < out_per_group_c)[None, :]
        & (h_out < out_h)[:, None]
        & (w_out < out_w)[:, None],
    )


class ConvTranspose2d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias, stride, padding, output_padding, groups, dilation
    ):
        logger.debug("GEMS CONV_TRANSPOSE2D")
        assert weight.ndim == 4, f"Weight must be 4D, received shape {weight.shape}"
        assert (
            bias is None or bias.ndim == 1
        ), f"Bias must be 1D, received shape {bias.shape}"
        assert (
            input.shape[1] == weight.shape[0]
        ), f"Incompatible input ({input.shape}) and weight ({weight.shape}) shapes"
        assert (
            bias is None or weight.shape[1] * groups == bias.shape[0]
        ), f"Incompatible weight ({weight.shape}) and bias ({bias.shape}) shapes"

        if isinstance(stride, (list, tuple)):
            stride_height, stride_width = stride
        else:
            stride_height = stride_width = stride

        if isinstance(padding, (list, tuple)):
            padding_height, padding_width = padding
        else:
            padding_height = padding_width = padding

        if isinstance(output_padding, (list, tuple)):
            output_padding_height, output_padding_width = output_padding
        else:
            output_padding_height = output_padding_width = output_padding

        if isinstance(dilation, (list, tuple)):
            dilation_height, dilation_width = dilation
        else:
            dilation_height = dilation_width = dilation

        in_n, in_c, input_height, input_width = input.shape
        out_per_group_c, weight_height, weight_width = (
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        )
        in_per_group_c = in_c // groups
        out_c = out_per_group_c * groups

        out_height = conv_transpose2d_output_size(
            input_height,
            weight_height,
            stride_height,
            padding_height,
            output_padding_height,
            dilation_height,
        )
        out_width = conv_transpose2d_output_size(
            input_width,
            weight_width,
            stride_width,
            padding_width,
            output_padding_width,
            dilation_width,
        )

        output_dtype = input.dtype

        # For fp32 stride=1, use conv2d with flipped+transposed weight.
        # conv2d's fp32 path is better optimized than our gather kernel.
        # For fp16/bf16 stride=1, use our phase-separated kernel (tensor cores friendly).
        use_conv2d_path = (
            stride_height == 1 and stride_width == 1 and output_dtype == torch.float32
        )

        if use_conv2d_path:
            w_flipped = torch.flip(weight, [2, 3])
            if groups == 1:
                w_conv2d = w_flipped.transpose(0, 1).contiguous()
            else:
                w_conv2d = (
                    w_flipped.view(
                        groups,
                        in_per_group_c,
                        out_per_group_c,
                        weight_height,
                        weight_width,
                    )
                    .transpose(1, 2)
                    .reshape(out_c, in_per_group_c, weight_height, weight_width)
                    .contiguous()
                )

            pad_h = dilation_height * (weight_height - 1) - padding_height
            pad_w = dilation_width * (weight_width - 1) - padding_width

            output = flag_gems_conv2d(
                input,
                w_conv2d,
                bias=bias,
                stride=1,
                padding=(pad_h, pad_w),
                dilation=(dilation_height, dilation_width),
                groups=groups,
            )
        else:
            output = torch.empty(
                (in_n, out_c, out_height, out_width),
                device=input.device,
                dtype=output_dtype,
            )

            if bias is None:
                bias_ptr = torch.zeros(out_c, device=input.device, dtype=output_dtype)
            else:
                bias_ptr = bias

            for phase_h in range(stride_height):
                for phase_w in range(stride_width):
                    h_per_phase = (
                        out_height - phase_h + stride_height - 1
                    ) // stride_height
                    w_per_phase = (
                        out_width - phase_w + stride_width - 1
                    ) // stride_width
                    if h_per_phase <= 0 or w_per_phase <= 0:
                        continue

                    grid = lambda META, h=h_per_phase, w=w_per_phase: (
                        triton.cdiv(in_n * h * w, META["BLOCK_NHW"]),
                        triton.cdiv(out_per_group_c, META["BLOCK_CO"]),
                        groups,
                    )

                    conv_transpose2d_forward_kernel[grid](
                        input,
                        weight,
                        output,
                        bias_ptr,
                        in_n,
                        input_height,
                        input_width,
                        out_height,
                        out_width,
                        h_per_phase,
                        w_per_phase,
                        *input.stride(),
                        *weight.stride(),
                        *output.stride(),
                        in_per_group_c,
                        out_per_group_c,
                        weight_height,
                        weight_width,
                        stride_height,
                        stride_width,
                        padding_height,
                        padding_width,
                        dilation_height,
                        dilation_width,
                        phase_h,
                        phase_w,
                        groups=groups,
                    )

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = (stride_height, stride_width)
        ctx.padding = (padding_height, padding_width)
        ctx.dilation = (dilation_height, dilation_width)
        ctx.groups = groups
        ctx.in_per_group_c = in_per_group_c
        ctx.out_per_group_c = out_per_group_c
        ctx.input_size = (in_n, input_height, input_width)
        ctx.out_size = (out_height, out_width)
        ctx.weight_size = (weight_height, weight_width)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS CONV_TRANSPOSE2D VJP")
        input, weight, bias = ctx.saved_tensors
        stride_height, stride_width = ctx.stride
        padding_height, padding_width = ctx.padding
        dilation_height, dilation_width = ctx.dilation
        groups = ctx.groups
        in_per_group_c = ctx.in_per_group_c
        out_per_group_c = ctx.out_per_group_c
        in_n, input_height, input_width = ctx.input_size
        out_height, out_width = ctx.out_size
        weight_height, weight_width = ctx.weight_size
        device = input.device

        grad_output = grad_output.contiguous()

        grad_input = torch.zeros(
            in_n,
            in_per_group_c * groups,
            input_height,
            input_width,
            dtype=grad_output.dtype,
            device=device,
        )
        bias_zero = torch.zeros(
            in_per_group_c * groups, device=device, dtype=grad_output.dtype
        )
        grid_input = lambda META: (
            triton.cdiv(in_n * input_height * input_width, META["BLOCK_NI_HO_WO"]),
            triton.cdiv(in_per_group_c, META["BLOCK_CO"]),
            groups,
        )
        conv2d_forward_kernel[grid_input](
            grad_output,
            weight,
            grad_input,
            bias_zero,
            in_n,
            out_height,
            out_width,
            in_per_group_c * groups,
            input_height,
            input_width,
            *grad_output.stride(),
            *weight.stride(),
            *grad_input.stride(),
            out_per_group_c,
            weight_height,
            weight_width,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
            groups=groups,
        )

        grad_weight = torch.zeros_like(weight)
        grid_weight = lambda meta: (
            triton.cdiv(
                out_per_group_c * weight_height * weight_width,
                meta["BLOCK_CI_HK_WK"],
            ),
            groups,
            triton.cdiv(in_per_group_c, meta["BLOCK_CO"]),
        )
        conv2d_backward_kernel_weight[grid_weight](
            grad_output,
            input,
            grad_weight,
            *grad_output.stride(),
            *grad_weight.stride(),
            *input.stride(),
            out_height,
            out_width,
            weight_height,
            weight_width,
            out_per_group_c,
            in_n,
            stride_height,
            stride_width,
            input_height,
            input_width,
            in_per_group_c,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
        )

        grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


def conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    return ConvTranspose2d.apply(
        input, weight, bias, stride, padding, output_padding, groups, dilation
    )
