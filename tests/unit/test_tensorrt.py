import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ntfields import Problem, TrainedField
from ntfields.envs import load_env
from ntfields.inference import autograd_inference
from ntfields.registry import create
from ntfields.speed import Clipped

pytest.importorskip("torch_tensorrt")
if not torch.cuda.is_available():
    pytest.skip("TensorRT needs CUDA", allow_module_level=True)

import tensorrt

from ntfields.accel.tensorrt import export

BATCH = (1, 64, 256)  # a narrow range builds faster


def _field(name: str) -> TrainedField:
    torch.manual_seed(0)
    method = create(name)
    problem = Problem(load_env("box2d/empty"), Clipped())
    return TrainedField(method, problem, method.build_model(problem).cuda().eval())


def _queries() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda").manual_seed(1)
    return (
        torch.rand(100, 2, device="cuda", generator=gen) - 0.5,
        torch.rand(100, 2, device="cuda", generator=gen) - 0.5,
    )


@pytest.mark.parametrize("name", ["td_ntfields", "pntfields"])
def test_fp32_engines_match_eager(name: str):
    field = _field(name)
    eager, fast = field.accelerate("eager", cuda_graphs=False), field.accelerate("tensorrt", batch=BATCH)
    q, y = _queries()
    goal = y[0]
    torch.testing.assert_close(
        fast.towards(goal).travel_time(q), eager.towards(goal).travel_time(q), rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(fast.towards(goal).gradient(q), eager.towards(goal).gradient(q), rtol=1e-4, atol=1e-4)
    for ours, expected in zip(fast.gradient(q, y), eager.gradient(q, y), strict=True):
        torch.testing.assert_close(ours, expected, rtol=1e-4, atol=1e-4)


def test_fp16_builds_only_the_travel_times_in_fp16():
    field = _field("td_ntfields")
    eager = field.accelerate("eager", cuda_graphs=False)
    fast = field.accelerate("tensorrt", precision="fp16", batch=BATCH)
    q, y = _queries()
    t, t_ref = fast.towards(y[0]).travel_time(q), eager.towards(y[0]).travel_time(q)
    assert (t - t_ref).abs().max() < 1e-2 * t_ref.abs().mean()
    # the goal encoding and the gradients are float32
    torch.testing.assert_close(fast.towards(y[0]).gradient(q), eager.towards(y[0]).gradient(q), rtol=1e-4, atol=1e-4)


def test_engines_are_built_on_first_call():
    fast = _field("td_ntfields").accelerate("tensorrt", cuda_graphs=False, batch=BATCH)
    q, y = _queries()
    fast.towards(y[0]).travel_time(q)
    built = {name: getattr(fast.inference, name)._engine is not None for name in ("travel_time", "gradient")}
    assert built == {"travel_time": True, "gradient": False}


def test_fp16_needs_float16_weights():
    with pytest.raises(ValueError, match="float16"):
        _field("pntfields").accelerate("tensorrt", precision="fp16")


def test_export_rejects_what_the_model_cannot_build(tmp_path: Path):
    with pytest.raises(ValueError, match="float16"):
        export(_field("pntfields"), tmp_path, precisions=("fp16",))
    with pytest.raises(ValueError, match="no inference model"):
        export(_field("euclidean"), tmp_path)


def test_exported_engines_reproduce_the_reference(tmp_path: Path):
    """The engines, run through the bare TensorRT runtime as another runtime would, on the exported reference."""
    field = _field("td_ntfields")
    functions = ("encode_goal", "travel_time", "gradient")
    export(field, tmp_path, functions, ("fp32", "fp16"), BATCH, n_ref=200)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["format_version"] == 1 and meta["dim"] == 2 and list(meta["engines"]) == list(functions)
    assert meta["engines"]["travel_time"]["precisions"] == ["fp32", "fp16"]
    assert meta["engines"]["gradient"]["precisions"] == meta["engines"]["encode_goal"]["precisions"] == ["fp32"]

    # the reference comes from the original model
    q = _load(tmp_path, "gradient", "q", ["B", 2], 200)
    e_g = _load(tmp_path, "gradient", "e_g", [1, -1], 200)
    reference = autograd_inference(field.model, field.problem.env.condition())
    assert reference.gradient is not None
    torch.testing.assert_close(_load(tmp_path, "gradient", "dt_dq", ["B", 2], 200), reference.gradient(q, e_g)[1])

    runtime = tensorrt.Runtime(tensorrt.Logger(tensorrt.Logger.ERROR))  # pyright: ignore[reportAttributeAccessIssue]
    for name, spec in meta["engines"].items():
        for precision in spec["precisions"]:
            engine = runtime.deserialize_cuda_engine((tmp_path / f"{name}_{precision}.engine").read_bytes())
            context = engine.create_execution_context()
            inputs = {b: _load(tmp_path, name, b, s, 200) for b, s in spec["inputs"].items()}
            expected = {b: _load(tmp_path, name, b, s, 200) for b, s in spec["outputs"].items()}
            outputs = {b: torch.empty_like(t) for b, t in expected.items()}
            for b, t in inputs.items():
                context.set_input_shape(b, tuple(t.shape))
            for b, t in {**inputs, **outputs}.items():
                context.set_tensor_address(b, t.data_ptr())
            stream = torch.cuda.Stream()
            assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            for b, t in outputs.items():
                # relative to the output's scale: the untrained field's gradients reach ~35 within the bounds, where
                # float32 rounding alone (the eager inference form's too) is ~1e-3
                tol = 1e-2 if precision == "fp16" else 1e-4
                scale = expected[b].abs().max().item()
                torch.testing.assert_close(t, expected[b], rtol=tol, atol=tol * scale, msg=f"{name}_{precision}: {b}")


def _load(out: Path, name: str, binding: str, shape: list[int | str], n: int) -> torch.Tensor:
    data = np.fromfile(out / "reference" / f"{name}_{binding}.bin", dtype=np.float32)
    return torch.from_numpy(data.reshape([n if d == "B" else int(d) for d in shape])).cuda()
