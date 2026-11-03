"""TensorRT builds of a model's inference form: in-process engines for `accelerate(field, "tensorrt")`, or serialized
engines with named bindings for other runtimes (`export`). Needs the [tensorrt] extra.

Every function is built with a dynamic batch B (min, opt, max) and float32 inputs and outputs; the bindings are
BINDINGS, with every input (B, dim) except the encoded goal e_g and encode_goal's q_g, both (1, ·):
- encode_goal: q_g (1, dim) -> e_g (1, n)
- travel_time: q (B, dim), e_g (1, n) -> t (B,)
- pair_travel_time: q_s, q_g (B, dim) -> t (B,)
- gradient: q (B, dim), e_g (1, n) -> t (B,), dt_dq (B, dim)
- pair_gradient: q_s, q_g (B, dim) -> t (B,), dt_dqs (B, dim), dt_dqg (B, dim)

The precision sets the weights' dtype of travel_time and pair_travel_time. encode_goal and the gradients are always
built in float32: encode_goal runs once per goal, and on the UR5e field an fp16 encoded goal alone puts an error of
8e-3 on dT/dq at B = 1024, where fp32 gives 3e-5 (|dT/dq| ~ 0.3). Float32 builds disable TF32, which on the UR5e
field puts errors of 8e-4 on T and 9e-3 on dT/dq at B = 1024, against 1e-6 and 3e-5 without. fp16 puts 4e-3 on T
(|T| ~ 1.3).
"""

import copy
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:
    import tensorrt
    import torch_tensorrt
except ImportError as e:
    raise ImportError("the TensorRT backend needs torch-tensorrt: pip install 'ntfields[tensorrt]'") from e
import torch
from torch import Tensor, nn

from ntfields.accel import Precision
from ntfields.field import TrainedField
from ntfields.inference import Function, InferenceModel, autograd_inference

FORMAT_VERSION = 1
Batch = tuple[int, int, int]  # (min, opt, max)
BATCH: Batch = (1, 128, 65536)
SETTINGS: dict[str, object] = {
    "min_block_size": 1,
    "require_full_compilation": True,
    "optimization_level": 5,
    "disable_tf32": True,
}
BINDINGS: dict[Function, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "encode_goal": (("q_g",), ("e_g",)),
    "travel_time": (("q", "e_g"), ("t",)),
    "pair_travel_time": (("q_s", "q_g"), ("t",)),
    "gradient": (("q", "e_g"), ("t", "dt_dq")),
    "pair_gradient": (("q_s", "q_g"), ("t", "dt_dqs", "dt_dqg")),
}
FLOAT32_ONLY: tuple[Function, ...] = ("encode_goal", "gradient", "pair_gradient")


def tensorrt_inference(
    inference: InferenceModel, precision: Precision = "fp32", batch: Batch = BATCH
) -> InferenceModel:
    """`inference` with each function built into a TensorRT engine on its first call."""
    _check(inference, precision)

    def engine(name: Function) -> Callable[..., Any]:
        return _LazyEngine(getattr(inference, name), name, precision, batch)

    return InferenceModel(
        engine("encode_goal"),
        engine("travel_time"),
        engine("pair_travel_time"),
        None if inference.gradient is None else engine("gradient"),
        None if inference.pair_gradient is None else engine("pair_gradient"),
        inference.float16_weights,
    )


def export(
    field: TrainedField,
    out: Path,
    functions: Sequence[Function] | None = None,
    precisions: Sequence[Precision] = ("fp32",),
    batch: Batch = BATCH,
    n_ref: int = 4096,
) -> None:
    """Serialized engines of `field`'s inference functions (default: all it has) for other runtimes, loadable with
    the same TensorRT version. Writes to `out`:
    - <function>_<precision>.engine, per precision for travel_time and pair_travel_time, and fp32 for the others
    - reference/<function>_<binding>.bin: the original model's inputs and outputs on n_ref random configurations
      within the env's bounds (raw little-endian float32), to check another runtime against
    - meta.json: the format version, dims, batch range, TensorRT version, the env's bounds and every engine's bindings
    """
    if field.device.type != "cuda":
        raise ValueError("TensorRT engines are built from a CUDA field")
    scene = field.problem.env.condition()
    inference = field.model.inference_model(scene)
    if inference is None:
        raise ValueError(f"{type(field.model).__name__} ({field.method.name}) has no inference model to export")
    for precision in precisions:
        _check(inference, precision)
    available: list[Function] = [name for name in BINDINGS if getattr(inference, name) is not None]
    names = list(functions) if functions else available
    if missing := [name for name in names if name not in available]:
        raise ValueError(f"{type(field.model).__name__}'s inference model has no {', '.join(missing)}")

    bounds = field.problem.env.bounds.to(field.device)
    gen = torch.Generator(device=field.device).manual_seed(0)
    q = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(
        n_ref, len(bounds), device=field.device, generator=gen
    )
    reference = autograd_inference(field.model, scene)
    with torch.no_grad():
        e_g = reference.encode_goal(q[:1])
    args: dict[Function, tuple[Tensor, ...]] = {
        "encode_goal": (q[:1],),
        "travel_time": (q, e_g),
        "pair_travel_time": (q, q.roll(1, dims=0)),
        "gradient": (q, e_g),
        "pair_gradient": (q, q.roll(1, dims=0)),
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "reference").mkdir(exist_ok=True)
    engines: dict[str, Any] = {}
    for name in names:
        inputs, outputs = BINDINGS[name]
        with torch.no_grad():
            expected = getattr(reference, name)(*args[name])
        expected = (expected,) if isinstance(expected, Tensor) else expected
        for binding, tensor in zip((*inputs, *outputs), (*args[name], *expected), strict=True):
            tensor.detach().float().cpu().numpy().tofile(out / "reference" / f"{name}_{binding}.bin")
        built = []
        chosen: list[Precision] = ["fp32"] if name in FLOAT32_ONLY else list(dict.fromkeys(precisions))
        for precision in chosen:
            module = _module(getattr(inference, name), name, precision)
            exported, specs = _export(module, name, args[name], batch)
            engine = torch_tensorrt.dynamo.convert_exported_program_to_serialized_trt_engine(
                exported,
                arg_inputs=specs,  # pyright: ignore[reportArgumentType]
                arg_input_binding_names=inputs,
                # a pytree matching the module's output: one name for a single tensor, a tuple for a tuple
                output_binding_names=outputs[0] if len(outputs) == 1 else outputs,
                **SETTINGS,  # type: ignore[arg-type]
            )
            (out / f"{name}_{precision}.engine").write_bytes(engine)
            built.append(precision)
        engines[name] = {
            "precisions": built,
            "inputs": {b: _shape(name, b, t) for b, t in zip(inputs, args[name], strict=True)},
            "outputs": {b: _shape(name, b, t) for b, t in zip(outputs, expected, strict=True)},
        }
    meta = {
        "format_version": FORMAT_VERSION,
        "method": field.method.name,
        "ckpt": None if field.ckpt_path is None else str(field.ckpt_path.resolve()),
        "dim": len(bounds),
        "encoded_goal": e_g.shape[1],
        "batch": batch,
        "tensorrt": tensorrt.__version__,
        "bounds": field.problem.env.bounds.tolist(),
        "n_ref": n_ref,
        "engines": engines,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


class _LazyEngine:
    """A function built into an in-process TensorRT engine on its first call (from that call's shapes, with the
    batch dynamic within `batch`), so functions that are never called are never built."""

    def __init__(self, fn: Callable[..., Any], name: Function, precision: Precision, batch: Batch) -> None:
        self.fn = fn
        self.name: Function = name
        self.precision: Precision = precision
        self.batch = batch
        self._engine: Callable[..., Any] | None = None

    def __call__(self, *args: Tensor) -> Any:
        if self._engine is None:
            module = _module(self.fn, self.name, self.precision)
            exported, specs = _export(module, self.name, args, self.batch)
            self._engine = torch_tensorrt.dynamo.compile(exported, arg_inputs=specs, **SETTINGS)  # type: ignore[arg-type]
        return self._engine(*args)


def _check(inference: InferenceModel, precision: Precision) -> None:
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"unknown precision {precision!r}; available: fp32, fp16")
    if precision == "fp16" and not inference.float16_weights:
        raise ValueError("this inference model does not support float16 weights; use precision='fp32'")


def _module(fn: Callable[..., Any], name: Function, precision: Precision) -> nn.Module:
    """A copy of the function's module, with float16 weights in fp16 unless the function is FLOAT32_ONLY."""
    if not isinstance(fn, nn.Module):
        raise TypeError(f"TensorRT builds nn.Modules; the inference model's {name} is a {type(fn).__name__}")
    dtype = torch.float16 if precision == "fp16" and name not in FLOAT32_ONLY else torch.float32
    return copy.deepcopy(fn).to(dtype).eval()


def _batched(name: Function, binding: str) -> bool:
    return name != "encode_goal" and binding != "e_g"


def _shape(name: Function, binding: str, example: Tensor) -> list[int | str]:
    return ["B", *example.shape[1:]] if _batched(name, binding) else list(example.shape)


def _export(
    module: nn.Module, name: Function, args: Sequence[Tensor], batch: Batch
) -> tuple[torch.export.ExportedProgram, list[torch_tensorrt.Input]]:
    """torch.export of `module`, dynamic in the batch unless min == max, and the matching TensorRT input specs."""
    batched = [_batched(name, binding) for binding in BINDINGS[name][0]]
    example = tuple(
        torch.zeros(batch[1], *a.shape[1:], device=a.device) if b else a.float()
        for a, b in zip(args, batched, strict=True)
    )
    specs = [
        torch_tensorrt.Input(
            min_shape=(batch[0], *a.shape[1:]), opt_shape=(batch[1], *a.shape[1:]), max_shape=(batch[2], *a.shape[1:])
        )
        if b
        else torch_tensorrt.Input(shape=tuple(a.shape))
        for a, b in zip(example, batched, strict=True)
    ]
    dynamic = None
    if batch[0] != batch[2]:
        dim = torch.export.Dim("batch", min=batch[0], max=batch[2])
        dynamic = tuple({0: dim} if b else None for b in batched)
    with torch.no_grad():
        return torch.export.export(module, example, dynamic_shapes=dynamic), specs
