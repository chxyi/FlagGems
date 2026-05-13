import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]
    MEDIAN_SHAPES = [(2, 32), (16, 64)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    INT_DTYPES = utils.INT_DTYPES
    DIM_LIST = [0, 1, -1]
    KEEPDIM = [True, False]
    MEDIAN_SHAPES = [
        (1, 2),
        (16, 128),
        (64, 256),
        (4096, 256),
        (64, 128, 32),
        (16, 64, 32, 8),
    ]


def _check_median_indices(inp, res_out, dim):
    """Verify that the indices returned by median point to the correct values."""
    indices = res_out.indices
    expected = res_out.values
    if indices.ndim < inp.ndim:
        indices = indices.unsqueeze(dim)
    gathered = inp.gather(dim, indices)
    if gathered.ndim > expected.ndim:
        gathered = gathered.squeeze(dim)
    assert torch.equal(
        gathered, expected
    ), f"Indices do not point to median values along dim={dim}"


@pytest.mark.median
@pytest.mark.parametrize("shape", MEDIAN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.median
@pytest.mark.parametrize("shape", MEDIAN_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_median_int(shape, dtype):
    inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.median
@pytest.mark.parametrize("shape", [(1,), (1, 1), (5, 1, 3)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_small(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.median_dim
@pytest.mark.parametrize("shape", MEDIAN_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim(shape, dim, keepdim, dtype):
    if dim >= len(shape):
        return
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(
        res_out.values, ref_out.values, dtype, reduce_dim=ref_out.values.numel()
    )
    _check_median_indices(inp, res_out, dim)


@pytest.mark.median_dim
@pytest.mark.parametrize("shape", MEDIAN_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_median_dim_int(shape, dim, keepdim, dtype):
    if dim >= len(shape):
        return
    inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out.values, ref_out.values)
    _check_median_indices(inp, res_out, dim)


@pytest.mark.median_dim
@pytest.mark.parametrize("shape", [(3, 5), (7, 11, 13)])
@pytest.mark.parametrize("keepdim, dim", [(True, -1), (False, 0)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_odd_size(shape, dim, keepdim, dtype):
    """Test median on odd-sized dimensions where the median is unambiguous."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(
        res_out.values, ref_out.values, dtype, reduce_dim=ref_out.values.numel()
    )
    _check_median_indices(inp, res_out, dim)


@pytest.mark.median_dim
@pytest.mark.parametrize("shape", [(4, 64), (8, 16, 64)])
@pytest.mark.parametrize("keepdim, dim", [(True, -1), (False, 0)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_even_size(shape, dim, keepdim, dtype):
    """Test median on even-sized dimensions (lower median is returned)."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_close(
        res_out.values, ref_out.values, dtype, reduce_dim=ref_out.values.numel()
    )
    _check_median_indices(inp, res_out, dim)
