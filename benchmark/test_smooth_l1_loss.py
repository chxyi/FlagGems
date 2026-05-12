import pytest
import torch

from . import base, consts


def _input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    target = torch.randn(shape, dtype=dtype, device=device)
    yield inp, target, 1, 1.0

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, target, 0, 1.0
        yield inp, target, 2, 1.0
        yield inp, target, 1, 0.5
        yield inp, target, 1, 2.0


@pytest.mark.smooth_l1_loss
def test_smooth_l1_loss():
    bench = base.GenericBenchmark(
        op_name="smooth_l1_loss",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.smooth_l1_loss,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
