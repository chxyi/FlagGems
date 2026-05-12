import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# ---------------------------------------------------------------------------
# Backward — basic
# ---------------------------------------------------------------------------


def _backward_helper(inp, target, reduction, beta):
    ref_inp = utils.to_reference(inp.detach().clone().requires_grad_())
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, beta)

    if reduction == 0:
        ref_grad = torch.ones_like(ref_out)
        ref_out.backward(ref_grad)
    else:
        ref_out.backward()

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, beta)
        if reduction == 0:
            res_grad = torch.ones_like(res_out)
            res_out.backward(res_grad)
        else:
            res_out.backward()

    return inp.grad, ref_inp.grad


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("shape", [(8, 16), (128, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_backward(shape, dtype, reduction):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    gems_grad, ref_grad = _backward_helper(inp, target, reduction, 1.0)
    utils.gems_assert_close(gems_grad, ref_grad, dtype)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("shape", [(4, 8), (64, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_backward_beta(shape, dtype):
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    for beta in [0.5, 2.0, 10.0]:
        inp = torch.randn(
            shape, dtype=dtype, device=flag_gems.device, requires_grad=True
        )
        gems_grad, ref_grad = _backward_helper(inp, target, 1, beta)
        utils.gems_assert_close(gems_grad, ref_grad, dtype)


@pytest.mark.smooth_l1_loss_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_backward_empty(dtype):
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device, requires_grad=True)
    target = torch.empty(0, dtype=dtype, device=flag_gems.device)
    gems_grad, ref_grad = _backward_helper(inp, target, 1, 1.0)
    utils.gems_assert_close(gems_grad, ref_grad, dtype)


# ---------------------------------------------------------------------------
# Backward — grad_input variant (out equivalent)
# ---------------------------------------------------------------------------


@pytest.mark.smooth_l1_loss_backward_out
@pytest.mark.parametrize("shape", [(8, 16), (128, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_backward_out(shape, dtype, reduction):
    if reduction == 0:
        grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        grad_output = torch.randn((), dtype=dtype, device=flag_gems.device)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_out = torch.ops.aten.smooth_l1_loss_backward(
        utils.to_reference(grad_output),
        utils.to_reference(inp),
        utils.to_reference(target),
        reduction,
        1.0,
    )

    grad_input = torch.empty_like(inp)
    with flag_gems.use_gems():
        res = torch.ops.aten.smooth_l1_loss_backward(
            grad_output, inp, target, reduction, 1.0, grad_input=grad_input
        )

    assert res is grad_input
    utils.gems_assert_close(grad_input, ref_out, dtype)
