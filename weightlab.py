#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

console = Console()
APP_HOME = Path(os.getenv("WEIGHTLAB_HOME", Path.home() / ".weightlab")).expanduser()
EXPERIMENTS_DIR = APP_HOME / "experiments"

TensorOperation = Callable[..., torch.Tensor]


# ---------------------------- utilities ----------------------------

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def slugify(value: str) -> str:
    value = value.strip().lower().replace("/", "--")
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    return re.sub(r"-+", "-", value).strip("-") or "experiment"

def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def save_json_atomic(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def get_git_commit() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return None

def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": get_git_commit(),
    }


# ---------------------------- operations ----------------------------

def op_identity(w: torch.Tensor) -> torch.Tensor:
    return w

def op_multiply(w: torch.Tensor, value: float) -> torch.Tensor:
    return w * value

def op_add(w: torch.Tensor, value: float) -> torch.Tensor:
    return w + value

def op_subtract(w: torch.Tensor, value: float) -> torch.Tensor:
    return w - value

def op_divide(w: torch.Tensor, value: float) -> torch.Tensor:
    if value == 0:
        raise ValueError("divide cannot use zero")
    return w / value

def op_l2_normalize(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = w.float()
    norm = torch.linalg.vector_norm(x)
    if float(norm) <= eps:
        return w
    return (x / norm).to(w.dtype)

def op_rms_normalize(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = w.float()
    rms = torch.sqrt(torch.mean(x * x))
    if float(rms) <= eps:
        return w
    return (x / rms).to(w.dtype)

def op_standardize(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = w.float()
    if x.numel() <= 1:
        return w
    std = x.std(unbiased=False)
    if float(std) <= eps:
        return w
    return ((x - x.mean()) / std).to(w.dtype)

def op_mean_center(w: torch.Tensor) -> torch.Tensor:
    x = w.float()
    return (x - x.mean()).to(w.dtype)

def op_minmax_normalize(
    w: torch.Tensor, min_value: float = 0.0, max_value: float = 1.0, eps: float = 1e-12
) -> torch.Tensor:
    x = w.float()
    lo, hi = x.min(), x.max()
    span = hi - lo
    if float(span) <= eps:
        return w
    y = (x - lo) / span
    return (y * (max_value - min_value) + min_value).to(w.dtype)

def op_maxabs_normalize(w: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = w.float()
    m = x.abs().max()
    if float(m) <= eps:
        return w
    return (x / m).to(w.dtype)

def op_clip(w: torch.Tensor, min: float, max: float) -> torch.Tensor:
    if min > max:
        raise ValueError("clip min cannot exceed max")
    return torch.clamp(w, min=min, max=max)

def op_round(w: torch.Tensor, decimals: int = 3) -> torch.Tensor:
    scale = 10.0 ** decimals
    return (torch.round(w.float() * scale) / scale).to(w.dtype)

def op_sign(w: torch.Tensor) -> torch.Tensor:
    return torch.sign(w)

def op_abs(w: torch.Tensor) -> torch.Tensor:
    return torch.abs(w)

def op_negate(w: torch.Tensor) -> torch.Tensor:
    return -w

def op_sin(w: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
    return torch.sin(w.float() * multiplier).to(w.dtype)

def op_cos(w: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
    return torch.cos(w.float() * multiplier).to(w.dtype)

def op_tanh(w: torch.Tensor, multiplier: float = 1.0) -> torch.Tensor:
    return torch.tanh(w.float() * multiplier).to(w.dtype)

def op_softsign(w: torch.Tensor) -> torch.Tensor:
    x = w.float()
    return (x / (1.0 + x.abs())).to(w.dtype)

def op_signed_power(w: torch.Tensor, exponent: float) -> torch.Tensor:
    x = w.float()
    return (torch.sign(x) * torch.pow(torch.abs(x), exponent)).to(w.dtype)

def op_gaussian_noise(
    w: torch.Tensor, std: float, mean: float = 0.0, seed: int | None = None
) -> torch.Tensor:
    if std < 0:
        raise ValueError("std must be >= 0")
    g = None
    if seed is not None:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
    noise = torch.normal(mean=mean, std=std, size=w.shape, generator=g, device=w.device, dtype=torch.float32)
    return (w.float() + noise).to(w.dtype)

def op_relative_gaussian_noise(w: torch.Tensor, std: float, seed: int | None = None) -> torch.Tensor:
    if std < 0:
        raise ValueError("std must be >= 0")
    g = None
    if seed is not None:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
    eps = torch.randn(w.shape, generator=g, device=w.device, dtype=torch.float32)
    x = w.float()
    return (x + std * x.abs() * eps).to(w.dtype)

def op_uniform_noise(
    w: torch.Tensor, low: float, high: float, seed: int | None = None
) -> torch.Tensor:
    if low > high:
        raise ValueError("low cannot exceed high")
    g = None
    if seed is not None:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
    noise = torch.empty(w.shape, dtype=torch.float32, device=w.device)
    noise.uniform_(low, high, generator=g)
    return (w.float() + noise).to(w.dtype)

def op_random_sign_flip(w: torch.Tensor, probability: float, seed: int | None = None) -> torch.Tensor:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    g = None
    if seed is not None:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
    mask = torch.rand(w.shape, generator=g, device=w.device) < probability
    return torch.where(mask, -w, w)

def op_magnitude_prune(w: torch.Tensor, threshold: float) -> torch.Tensor:
    x = w.clone()
    x[x.abs() < threshold] = 0
    return x

def op_keep_top_percent(w: torch.Tensor, percent: float) -> torch.Tensor:
    if not 0 < percent <= 100:
        raise ValueError("percent must be > 0 and <= 100")
    if percent == 100 or w.numel() == 0:
        return w
    flat = w.abs().flatten().float()
    k = max(1, math.ceil(flat.numel() * percent / 100.0))
    threshold = torch.topk(flat, k, largest=True, sorted=False).values.min()
    return torch.where(w.abs() >= threshold.to(w.dtype), w, torch.zeros_like(w))

def op_random_prune(w: torch.Tensor, probability: float, seed: int | None = None) -> torch.Tensor:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    g = None
    if seed is not None:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
    keep = torch.rand(w.shape, generator=g, device=w.device) >= probability
    return torch.where(keep, w, torch.zeros_like(w))

def op_shuffle_weights(w: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    g = None
    if seed is not None:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
    flat = w.flatten()
    perm = torch.randperm(flat.numel(), generator=g, device=w.device)
    return flat[perm].reshape_as(w)

def op_shuffle_magnitudes(w: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    g = None
    if seed is not None:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
    flat = w.flatten()
    mags = flat.abs()
    perm = torch.randperm(flat.numel(), generator=g, device=w.device)
    return (torch.sign(flat) * mags[perm]).reshape_as(w)

def op_shuffle_signs(w: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    g = None
    if seed is not None:
        g = torch.Generator(device=w.device)
        g.manual_seed(seed)
    flat = w.flatten()
    signs = torch.sign(flat)
    perm = torch.randperm(flat.numel(), generator=g, device=w.device)
    return (flat.abs() * signs[perm]).reshape_as(w)

def op_interpolate(w: torch.Tensor, alpha: float, target: str = "zero") -> torch.Tensor:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if target == "zero":
        target_tensor = torch.zeros_like(w)
    elif target == "sign":
        target_tensor = torch.sign(w)
    elif target == "l2_normalized":
        target_tensor = op_l2_normalize(w)
    else:
        raise ValueError("target must be one of: zero, sign, l2_normalized")
    return ((1 - alpha) * w.float() + alpha * target_tensor.float()).to(w.dtype)

def op_simulated_symmetric_quantize(w: torch.Tensor, bits: int = 8, eps: float = 1e-12) -> torch.Tensor:
    if bits < 2 or bits > 16:
        raise ValueError("bits must be between 2 and 16")
    x = w.float()
    qmax = (2 ** (bits - 1)) - 1
    m = x.abs().max()
    if float(m) <= eps:
        return w
    scale = m / qmax
    q = torch.clamp(torch.round(x / scale), -qmax, qmax)
    return (q * scale).to(w.dtype)

OPERATIONS: dict[str, TensorOperation] = {
    "identity": op_identity,
    "multiply": op_multiply,
    "add": op_add,
    "subtract": op_subtract,
    "divide": op_divide,
    "l2_normalize": op_l2_normalize,
    "rms_normalize": op_rms_normalize,
    "standardize": op_standardize,
    "mean_center": op_mean_center,
    "minmax_normalize": op_minmax_normalize,
    "maxabs_normalize": op_maxabs_normalize,
    "clip": op_clip,
    "round": op_round,
    "sign": op_sign,
    "abs": op_abs,
    "negate": op_negate,
    "sin": op_sin,
    "cos": op_cos,
    "tanh": op_tanh,
    "softsign": op_softsign,
    "signed_power": op_signed_power,
    "gaussian_noise": op_gaussian_noise,
    "relative_gaussian_noise": op_relative_gaussian_noise,
    "uniform_noise": op_uniform_noise,
    "random_sign_flip": op_random_sign_flip,
    "magnitude_prune": op_magnitude_prune,
    "keep_top_percent": op_keep_top_percent,
    "random_prune": op_random_prune,
    "shuffle_weights": op_shuffle_weights,
    "shuffle_magnitudes": op_shuffle_magnitudes,
    "shuffle_signs": op_shuffle_signs,
    "interpolate": op_interpolate,
    "quantize": op_simulated_symmetric_quantize,
}


# ---------------------------- pipeline ----------------------------

@dataclass(frozen=True)
class PipelineStep:
    operation: str
    parameters: dict[str, Any]

def apply_pipeline(tensor: torch.Tensor, pipeline: list[PipelineStep]) -> torch.Tensor:
    result = tensor
    original_shape = tensor.shape
    original_dtype = tensor.dtype

    for i, step in enumerate(pipeline, 1):
        result = OPERATIONS[step.operation](result, **step.parameters)
        if not isinstance(result, torch.Tensor):
            raise TypeError(f"Step {i} returned {type(result).__name__}, expected Tensor")
        if result.shape != original_shape:
            raise ValueError(f"Step {i} changed shape {tuple(original_shape)} -> {tuple(result.shape)}")
        if not torch.isfinite(result.float()).all():
            raise ValueError(f"Step {i} ({step.operation}) generated NaN or Inf")
    return result.to(dtype=original_dtype, device="cpu").contiguous()

def pipeline_to_dict(pipeline: list[PipelineStep]) -> list[dict[str, Any]]:
    return [
        {"order": i, "operation": s.operation, "parameters": s.parameters}
        for i, s in enumerate(pipeline, 1)
    ]

def print_pipeline(pipeline: list[PipelineStep]) -> None:
    table = Table(title="Current Pipeline")
    table.add_column("#", justify="right")
    table.add_column("Operation")
    table.add_column("Parameters")
    if not pipeline:
        table.add_row("-", "Original weights", "-")
    else:
        for i, s in enumerate(pipeline, 1):
            table.add_row(str(i), s.operation, json.dumps(s.parameters, ensure_ascii=False))
    console.print(table)

def choose_operation() -> PipelineStep | None:
    options = [
        ("identity", "Identity"),
        ("multiply", "Multiply"),
        ("add", "Add"),
        ("subtract", "Subtract"),
        ("divide", "Divide"),
        ("l2_normalize", "L2 normalize"),
        ("rms_normalize", "RMS normalize"),
        ("standardize", "Standardize"),
        ("mean_center", "Mean center"),
        ("minmax_normalize", "Min-Max normalize"),
        ("maxabs_normalize", "Max-Abs normalize"),
        ("clip", "Clip"),
        ("round", "Round"),
        ("sign", "Sign only"),
        ("abs", "Absolute value"),
        ("negate", "Negate"),
        ("sin", "Sine"),
        ("cos", "Cosine"),
        ("tanh", "Tanh"),
        ("softsign", "Softsign"),
        ("signed_power", "Signed power"),
        ("gaussian_noise", "Gaussian noise"),
        ("relative_gaussian_noise", "Relative Gaussian noise"),
        ("uniform_noise", "Uniform noise"),
        ("random_sign_flip", "Random sign flip"),
        ("magnitude_prune", "Magnitude prune"),
        ("keep_top_percent", "Keep top magnitude %"),
        ("random_prune", "Random prune"),
        ("shuffle_weights", "Shuffle weights"),
        ("shuffle_magnitudes", "Shuffle magnitudes"),
        ("shuffle_signs", "Shuffle signs"),
        ("interpolate", "Interpolate"),
        ("quantize", "Simulated symmetric quantization"),
    ]

    table = Table(title="Choose Operation")
    table.add_column("#", justify="right")
    table.add_column("Operation")
    for i, (_, label) in enumerate(options, 1):
        table.add_row(str(i), label)
    table.add_row("0", "Cancel")
    console.print(table)

    idx = IntPrompt.ask("Selection", default=0)
    if idx == 0:
        return None
    if not 1 <= idx <= len(options):
        console.print("[red]Invalid selection[/red]")
        return None

    name = options[idx - 1][0]
    p: dict[str, Any] = {}

    if name in {"multiply", "add", "subtract", "divide"}:
        special = Prompt.ask("Value (number, pi, e)", default="pi" if name == "multiply" else "1")
        if special.lower() == "pi":
            p["value"] = math.pi
        elif special.lower() == "e":
            p["value"] = math.e
        else:
            p["value"] = float(special)
    elif name == "minmax_normalize":
        p["min_value"] = FloatPrompt.ask("Target min", default=0.0)
        p["max_value"] = FloatPrompt.ask("Target max", default=1.0)
    elif name == "clip":
        p["min"] = FloatPrompt.ask("Min", default=-0.1)
        p["max"] = FloatPrompt.ask("Max", default=0.1)
    elif name == "round":
        p["decimals"] = IntPrompt.ask("Decimals", default=3)
    elif name in {"sin", "cos", "tanh"}:
        p["multiplier"] = FloatPrompt.ask("Multiplier", default=1.0)
    elif name == "signed_power":
        p["exponent"] = FloatPrompt.ask("Exponent", default=2.0)
    elif name == "gaussian_noise":
        p["mean"] = FloatPrompt.ask("Mean", default=0.0)
        p["std"] = FloatPrompt.ask("Std", default=0.001)
        p["seed"] = IntPrompt.ask("Seed", default=42)
    elif name == "relative_gaussian_noise":
        p["std"] = FloatPrompt.ask("Relative std", default=0.001)
        p["seed"] = IntPrompt.ask("Seed", default=42)
    elif name == "uniform_noise":
        p["low"] = FloatPrompt.ask("Low", default=-0.001)
        p["high"] = FloatPrompt.ask("High", default=0.001)
        p["seed"] = IntPrompt.ask("Seed", default=42)
    elif name in {"random_sign_flip", "random_prune"}:
        p["probability"] = FloatPrompt.ask("Probability [0..1]", default=0.01)
        p["seed"] = IntPrompt.ask("Seed", default=42)
    elif name == "magnitude_prune":
        p["threshold"] = FloatPrompt.ask("Magnitude threshold", default=0.001)
    elif name == "keep_top_percent":
        p["percent"] = FloatPrompt.ask("Percent to retain", default=90.0)
    elif name in {"shuffle_weights", "shuffle_magnitudes", "shuffle_signs"}:
        p["seed"] = IntPrompt.ask("Seed", default=42)
    elif name == "interpolate":
        p["alpha"] = FloatPrompt.ask("Alpha [0..1]", default=0.5)
        p["target"] = Prompt.ask("Target", choices=["zero", "sign", "l2_normalized"], default="zero")
    elif name == "quantize":
        p["bits"] = IntPrompt.ask("Bits", default=8)

    return PipelineStep(name, p)

def build_pipeline_interactively() -> list[PipelineStep]:
    pipeline: list[PipelineStep] = []

    while True:
        print_pipeline(pipeline)
        console.print("[bold]A[/bold] Add  [bold]D[/bold] Delete  [bold]M[/bold] Move  [bold]R[/bold] Run  [bold]Q[/bold] Cancel")
        cmd = Prompt.ask("Command", default="A").strip().lower()

        if cmd == "a":
            step = choose_operation()
            if step:
                pipeline.append(step)
        elif cmd == "d":
            if not pipeline:
                continue
            idx = IntPrompt.ask("Step to delete")
            if 1 <= idx <= len(pipeline):
                pipeline.pop(idx - 1)
        elif cmd == "m":
            if len(pipeline) < 2:
                continue
            src = IntPrompt.ask("Move step")
            dst = IntPrompt.ask("To position")
            if 1 <= src <= len(pipeline) and 1 <= dst <= len(pipeline):
                step = pipeline.pop(src - 1)
                pipeline.insert(dst - 1, step)
        elif cmd == "r":
            if pipeline:
                return pipeline
            console.print("[yellow]Add at least one operation.[/yellow]")
        elif cmd == "q":
            raise KeyboardInterrupt


# ---------------------------- checkpoint handling ----------------------------

def discover_shards(source_dir: Path) -> list[Path]:
    index = source_dir / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(data.get("weight_map", {}).values()))
        shards = [source_dir / n for n in names]
    elif (source_dir / "model.safetensors").exists():
        shards = [source_dir / "model.safetensors"]
    else:
        shards = sorted(source_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError("No .safetensors checkpoint found")
    for p in shards:
        if not p.exists():
            raise FileNotFoundError(f"Missing shard: {p}")
    return shards

def copy_support_files(source: Path, dest: Path) -> list[str]:
    copied = []
    for item in source.iterdir():
        if not item.is_file() or item.suffix == ".safetensors":
            continue
        if item.name in {"manifest.json", "transformation.log", "pipeline.yaml"}:
            continue
        shutil.copy2(item, dest / item.name)
        copied.append(item.name)
    return sorted(copied)

def transform_shard(source: Path, dest: Path, pipeline: list[PipelineStep]) -> dict[str, Any]:
    started = time.perf_counter()
    source_hash = sha256_file(source)

    with safe_open(source, framework="pt", device="cpu") as f:
        names = list(f.keys())
        metadata = f.metadata()
        sizes = {n: math.prod(f.get_slice(n).get_shape()) for n in names}
        total = sum(sizes.values())

    out: dict[str, torch.Tensor] = {}
    floating_params = 0
    floating_tensors = 0

    progress = tqdm(total=total, desc=source.name, unit="param", unit_scale=True, dynamic_ncols=True)
    try:
        with safe_open(source, framework="pt", device="cpu") as f:
            for name in names:
                tensor = f.get_tensor(name)
                if tensor.is_floating_point():
                    result = apply_pipeline(tensor, pipeline)
                    out[name] = result
                    floating_params += result.numel()
                    floating_tensors += 1
                    del result
                else:
                    out[name] = tensor.contiguous()
                progress.update(sizes[name])
                del tensor
    finally:
        progress.close()

    tmp = dest.with_name(dest.name + ".tmp")
    save_file(out, tmp, metadata=metadata)
    tmp.replace(dest)
    output_hash = sha256_file(dest)
    del out
    gc.collect()

    return {
        "source_file": source.name,
        "output_file": dest.name,
        "source_sha256": source_hash,
        "output_sha256": output_hash,
        "total_parameters": total,
        "floating_parameters_transformed": floating_params,
        "floating_tensors_transformed": floating_tensors,
        "duration_seconds": time.perf_counter() - started,
    }

def validate_checkpoint(source_shards: list[Path], output_dir: Path) -> dict[str, Any]:
    src = {}
    dst = {}
    for shard in source_shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for n in f.keys():
                src[n] = (tuple(f.get_slice(n).get_shape()), str(f.get_tensor(n).dtype))
    for shard in source_shards:
        op = output_dir / shard.name
        if not op.exists():
            raise FileNotFoundError(f"Missing output shard: {op}")
        with safe_open(op, framework="pt", device="cpu") as f:
            for n in f.keys():
                dst[n] = (tuple(f.get_slice(n).get_shape()), str(f.get_tensor(n).dtype))
    if src != dst:
        raise RuntimeError("Checkpoint structure or dtype differs after transformation")
    return {"valid": True, "tensor_count": len(src)}


# ---------------------------- interactive model flow ----------------------------

def resolve_model(repo_id: str, revision: str) -> tuple[Path, str]:
    console.print(f"\nResolving [cyan]{repo_id}[/cyan] @ [cyan]{revision}[/cyan]...")
    api = HfApi()
    info = api.model_info(repo_id, revision=revision)
    resolved = info.sha
    console.print(f"Resolved commit: [dim]{resolved}[/dim]")
    path = snapshot_download(repo_id=repo_id, revision=resolved)
    return Path(path), resolved

def suggest_experiment_name(pipeline: list[PipelineStep]) -> str:
    parts = []
    for s in pipeline[:4]:
        name = s.operation.replace("_", "-")
        if s.operation == "multiply" and abs(float(s.parameters.get("value", 0)) - math.pi) < 1e-12:
            name = "pi"
        parts.append(name)
    return slugify("-".join(parts))[:80]

def save_pipeline_yaml(path: Path, model: dict[str, Any], pipeline: list[PipelineStep], name: str) -> None:
    data = {
        "version": 1,
        "name": name,
        "model": model,
        "pipeline": [{"operation": s.operation, **s.parameters} for s in pipeline],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def run_experiment(repo_id: str, revision: str, pipeline: list[PipelineStep], name: str | None = None) -> Path:
    source, resolved = resolve_model(repo_id, revision)
    shards = discover_shards(source)

    exp_name = name or suggest_experiment_name(pipeline)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = EXPERIMENTS_DIR / slugify(repo_id)
    out_dir = base / f"{stamp}-{slugify(exp_name)}"
    model_dir = out_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=False)

    manifest_path = out_dir / "manifest.json"
    pipeline_path = out_dir / "pipeline.yaml"
    log_path = out_dir / "transformation.log"

    model_meta = {"repo_id": repo_id, "requested_revision": revision, "resolved_revision": resolved}
    save_pipeline_yaml(pipeline_path, model_meta, pipeline, exp_name)

    manifest = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "status": "running",
        "started_at": utc_now(),
        "model": model_meta,
        "experiment_name": exp_name,
        "pipeline": pipeline_to_dict(pipeline),
        "pipeline_sha256": canonical_hash(pipeline_to_dict(pipeline)),
        "environment": environment_info(),
        "shards": [],
    }
    save_json_atomic(manifest_path, manifest)

    log = log_path.open("a", encoding="utf-8")
    def logline(msg: str):
        line = f"{utc_now()} | {msg}"
        console.print(line)
        log.write(line + "\n")
        log.flush()

    try:
        logline(f"Source cache: {source}")
        logline(f"Output: {out_dir}")
        logline(f"Shards: {len(shards)}")
        manifest["copied_supporting_files"] = copy_support_files(source, model_dir)

        for shard in tqdm(shards, desc="Checkpoint", unit="shard", dynamic_ncols=True):
            result = transform_shard(shard, model_dir / shard.name, pipeline)
            manifest["shards"].append(result)
            save_json_atomic(manifest_path, manifest)

        validation = validate_checkpoint(shards, model_dir)
        manifest["validation"] = validation
        manifest["status"] = "success"
        manifest["finished_at"] = utc_now()
        save_json_atomic(manifest_path, manifest)
        logline("Experiment completed successfully.")
        return out_dir

    except Exception as e:
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        save_json_atomic(manifest_path, manifest)
        logline(f"FAILED: {e}")
        raise
    finally:
        log.close()

def list_experiments() -> list[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        [p for p in EXPERIMENTS_DIR.glob("*/*") if (p / "manifest.json").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

def interactive() -> None:
    console.print(Panel.fit(
        "[bold cyan]WeightLab[/bold cyan]\n"
        "Interactive neural-weight transformation laboratory",
        border_style="cyan",
    ))
    repo_id = Prompt.ask("Hugging Face model", default="google/gemma-3-1b-it")
    revision = Prompt.ask("Revision", default="main")
    pipeline = build_pipeline_interactively()
    print_pipeline(pipeline)

    suggested = suggest_experiment_name(pipeline)
    name = Prompt.ask("Experiment name", default=suggested)

    if not Confirm.ask("Run experiment now?", default=True):
        return

    out = run_experiment(repo_id, revision, pipeline, name)
    console.print(Panel.fit(
        f"[green]Experiment complete[/green]\n{out}\n\n"
        f"Run [bold]python test_model.py[/bold] to compare it.",
        border_style="green",
    ))

def main() -> int:
    parser = argparse.ArgumentParser(description="WeightLab interactive model-weight experimentation")
    parser.add_argument("--model")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    try:
        if args.model:
            repo_id = args.model
            pipeline = build_pipeline_interactively()
            name = Prompt.ask("Experiment name", default=suggest_experiment_name(pipeline))
            out = run_experiment(repo_id, args.revision, pipeline, name)
            console.print(f"[green]Saved:[/green] {out}")
        else:
            interactive()
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return 130
    except Exception:
        console.print_exception()
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
