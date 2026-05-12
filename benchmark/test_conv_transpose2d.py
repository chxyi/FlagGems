from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts

SHAPES = [
    # (batch, in_c, in_h, in_w, out_c_per_group, kernel_h, kernel_w, stride, padding, groups)
    (32, 64, 16, 16, 32, 3, 3, 1, 1, 1),
    (32, 64, 16, 16, 32, 3, 3, 2, 1, 1),
    (16, 32, 8, 8, 16, 4, 4, 2, 1, 1),
    (16, 32, 12, 12, 32, 3, 3, 2, 1, 2),
    (8, 64, 32, 32, 32, 5, 5, 1, 2, 1),
    (4, 128, 16, 16, 64, 3, 3, 2, 1, 1),
]


class ConvTranspose2DBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


def _input_fn(shape, dtype, device):
    (
        batch,
        in_c,
        in_h,
        in_w,
        out_c_per_group,
        kernel_h,
        kernel_w,
        stride,
        padding,
        groups,
    ) = shape
    # weight shape for conv_transpose2d: [C_in, C_out/groups, kH, kW]
    input_shape = (batch, in_c, in_h, in_w)
    weight_shape = (in_c, out_c_per_group, kernel_h, kernel_w)
    inp = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    yield {
        "input": inp,
        "weight": weight,
        "bias": None,
        "stride": stride,
        "padding": padding,
        "groups": groups,
    },


@pytest.mark.conv_transpose2d
def test_conv_transpose2d(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBenchmark(
        input_fn=_input_fn,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv_transpose2d)

    bench.run()
