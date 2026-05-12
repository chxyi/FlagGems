import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

vendor_name = flag_gems.vendor_name

# (input_shape, weight_shape, groups)
# weight shape for conv_transpose2d: [C_in, C_out/groups, kH, kW]
SHAPE_CONV_TRANSPOSE2D = [
    # small
    ((1, 2, 3, 3), (2, 1, 2, 2), 1),
    ((2, 4, 5, 5), (4, 2, 3, 3), 1),
    # regular
    ((4, 8, 8, 8), (8, 8, 3, 3), 1),
    ((2, 4, 16, 16), (4, 4, 4, 4), 1),
    # groups
    ((2, 4, 8, 8), (4, 2, 3, 3), 2),
    ((2, 8, 8, 8), (8, 2, 3, 3), 4),
    # large
    ((1, 4, 32, 32), (4, 4, 5, 5), 1),
    ((2, 8, 64, 64), (8, 4, 3, 3), 1),
]


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("shape, kernel, groups", SHAPE_CONV_TRANSPOSE2D)
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv_transpose2d(
    monkeypatch, shape, kernel, stride, padding, groups, dtype, dilation, bias
):
    if vendor_name == "mthreads" and dtype == torch.float16:
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    if vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp, True)

    torch.backends.cudnn.allow_tf32 = False

    weight = torch.randn(
        kernel, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    if bias is True:
        bias_tensor = torch.randn(
            [weight.shape[1] * groups],
            dtype=dtype,
            device=flag_gems.device,
            requires_grad=True,
        )
        bias_ref = utils.to_reference(bias_tensor, True)
    else:
        bias_tensor = None
        bias_ref = None

    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.nn.functional.conv_transpose2d(
        ref_inp,
        ref_weight,
        bias=bias_ref,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)

    res_out = flag_gems.conv_transpose2d(
        inp,
        weight,
        bias=bias_tensor,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)

    out_grad = torch.randn_like(ref_out).to(flag_gems.device)

    ref_grad = utils.to_reference(out_grad, True)
    if bias is not None:
        ref_in_grad, ref_weight_grad, ref_bias_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, bias_ref), ref_grad
        )
        res_in_grad, res_weight_grad, res_bias_grad = torch.autograd.grad(
            res_out, (inp, weight, bias_tensor), out_grad
        )
    else:
        ref_in_grad, ref_weight_grad = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight), ref_grad
        )
        res_in_grad, res_weight_grad = torch.autograd.grad(
            res_out, (inp, weight), out_grad
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=kernel[2])

    utils.gems_assert_close(
        res_weight_grad, ref_weight_grad, dtype, reduce_dim=shape[0] * shape[2]
    )
    if bias is not None:
        utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_transpose2d_output_padding(monkeypatch, dtype):
    """Test output_padding > 0 to produce different output sizes."""
    if vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(1, 2, 4, 4, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(2, 2, 3, 3, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)

    for op in [0, 1]:
        ref_out = torch.nn.functional.conv_transpose2d(
            ref_inp, ref_weight, stride=2, output_padding=op
        ).to(dtype)
        res_out = flag_gems.conv_transpose2d(inp, weight, stride=2, output_padding=op)
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.conv_transpose2d
@pytest.mark.parametrize("dtype", [torch.float32])
def test_conv_transpose2d_small(monkeypatch, dtype):
    """Test 1x1 spatial input and 1x1 kernel."""
    if vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False

    inp = torch.randn(1, 1, 1, 1, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    weight = torch.randn(1, 1, 1, 1, dtype=dtype, device=flag_gems.device)
    ref_weight = utils.to_reference(weight, True)
    ref_out = torch.nn.functional.conv_transpose2d(ref_inp, ref_weight).to(dtype)
    res_out = flag_gems.conv_transpose2d(inp, weight)
    utils.gems_assert_close(res_out, ref_out, dtype)
