import pytest
import torch

from . import base, consts


def _backward_input_fn(shape, dtype, device):
    # Generate tensors suitable for directly calling smooth_l1_loss_backward.
    # grad_output is a scalar (mean/sum) or full tensor (none).
    grad_output = torch.randn(shape, dtype=dtype, device=device)
    x = torch.randn(shape, dtype=dtype, device=device)
    y = torch.randn(shape, dtype=dtype, device=device)
    yield grad_output, x, y, 0, 1.0

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield grad_output, x, y, 1, 1.0
        yield grad_output, x, y, 2, 1.0
        # For mean/sum: use scalar grad_output (reduction 1)
        yield torch.randn((), dtype=dtype, device=device), x, y, 1, 1.0
        yield torch.randn((), dtype=dtype, device=device), x, y, 2, 1.0


@pytest.mark.smooth_l1_loss_backward
def test_smooth_l1_loss_backward():
    bench = base.GenericBenchmark(
        op_name="smooth_l1_loss_backward",
        input_fn=_backward_input_fn,
        torch_op=torch.ops.aten.smooth_l1_loss_backward,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=False,
    )
    bench.run()
