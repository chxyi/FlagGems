import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# ---------------------------------------------------------------------------
# Forward — basic
# ---------------------------------------------------------------------------


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_smooth_l1_loss(shape, dtype, reduction, beta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, float(beta))

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, float(beta))

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize(
    "shape_pair",
    [
        ((2, 3), (2, 3)),
        ((1, 256), (128, 256)),
        ((128, 1), (128, 256)),
        ((20, 1), (20, 320)),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_broadcast(shape_pair, dtype, reduction):
    inp_shape, target_shape = shape_pair
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(target_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_empty(dtype):
    inp = torch.empty(0, dtype=dtype, device=flag_gems.device)
    target = torch.empty(0, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)

    for reduction in [0, 1, 2]:
        ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0)
        with flag_gems.use_gems():
            res_out = torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0)
        if reduction == 1:
            utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
        else:
            utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_large_beta(dtype):
    shape = (16, 32)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, 1, 10.0)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, 1, 10.0)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_small_beta(dtype):
    shape = (16, 32)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.01
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 0.01

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    ref_out = torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, 1, 0.01)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.smooth_l1_loss(inp, target, 1, 0.01)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Forward — out variant
# ---------------------------------------------------------------------------


@pytest.mark.smooth_l1_loss_out
@pytest.mark.parametrize("shape", [(4, 8), (128, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_smooth_l1_loss_out(shape, dtype, reduction):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    if reduction == 0:
        out = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        out = torch.empty((), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_target = utils.to_reference(target)
    if reduction == 0:
        ref_out = torch.empty(shape, dtype=dtype, device=ref_inp.device)
    else:
        ref_out = torch.empty((), dtype=dtype, device=ref_inp.device)
    torch.ops.aten.smooth_l1_loss(ref_inp, ref_target, reduction, 1.0, out=ref_out)

    with flag_gems.use_gems():
        torch.ops.aten.smooth_l1_loss(inp, target, reduction, 1.0, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.smooth_l1_loss_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_smooth_l1_loss_out_preallocated(dtype):
    shape = (4, 8)
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res1 = torch.ops.aten.smooth_l1_loss(inp1, target1, 0, 1.0, out=out)
    assert res1 is out

    with flag_gems.use_gems():
        res2 = torch.ops.aten.smooth_l1_loss(inp2, target2, 0, 1.0, out=out)
    assert res2 is out

    ref_out = torch.ops.aten.smooth_l1_loss(
        utils.to_reference(inp2), utils.to_reference(target2), 0, 1.0
    )
    utils.gems_assert_close(out, ref_out, dtype)
