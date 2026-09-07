#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()
APP_HOME = Path(os.getenv("WEIGHTLAB_HOME", Path.home() / ".weightlab")).expanduser()
EXPERIMENTS_DIR = APP_HOME / "experiments"

DEFAULT_PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "What is 2 + 2?",
    "Complete the sentence: The sky is",
    "Write a Python function that takes two integers and returns their sum.",
    "Alice has 5 apples and gives Bob 2 apples. How many apples does Alice have left?",
    "Explain the difference between RAM and disk storage in simple terms.",
]

def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def list_experiments() -> list[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []
    items = [p for p in EXPERIMENTS_DIR.glob("*/*") if (p / "manifest.json").exists()]
    return sorted(items, key=lambda p: p.stat().st_mtime, reverse=True)

def choose_experiment() -> Path:
    exps = list_experiments()
    if not exps:
        raise FileNotFoundError(f"No experiments found under {EXPERIMENTS_DIR}")

    table = Table(title="Saved Experiments")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Experiment")
    table.add_column("Status")
    for i, p in enumerate(exps, 1):
        m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        table.add_row(
            str(i),
            m.get("model", {}).get("repo_id", "?"),
            m.get("experiment_name", p.name),
            m.get("status", "?"),
        )
    console.print(table)
    idx = IntPrompt.ask("Choose experiment", default=1)
    if not 1 <= idx <= len(exps):
        raise ValueError("Invalid selection")
    return exps[idx - 1]

def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("Prompt JSON must be an array of strings")
        return data
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def format_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt

def evaluate(model_path: str, prompts: list[str], device: str, max_new_tokens: int) -> dict[str, Any]:
    console.print(f"\n[cyan]Loading[/cyan] {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    rows = []
    for prompt in tqdm(prompts, desc="Generating", unit="prompt", dynamic_ncols=True):
        formatted = format_prompt(tokenizer, prompt)
        enc = tokenizer(formatted, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        input_tokens = int(enc["input_ids"].shape[-1])

        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        sync(device)

        elapsed = time.perf_counter() - start
        generated = out[0, input_tokens:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        count = int(generated.shape[-1])

        rows.append({
            "prompt": prompt,
            "response": text,
            "input_tokens": input_tokens,
            "generated_tokens": count,
            "generation_seconds": elapsed,
            "tokens_per_second": count / elapsed if elapsed else 0.0,
        })

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"model": model_path, "device": device, "results": rows}

def print_comparison(a: dict[str, Any], b: dict[str, Any]) -> None:
    console.print("\n[bold]Comparison[/bold]")
    for i, (x, y) in enumerate(zip(a["results"], b["results"]), 1):
        console.print(Panel.fit(
            f"[bold]Prompt {i}[/bold]\n{x['prompt']}\n\n"
            f"[cyan]Original[/cyan]\n{x['response']}\n\n"
            f"[magenta]Transformed[/magenta]\n{y['response']}\n\n"
            f"Speed: {x['tokens_per_second']:.2f} vs {y['tokens_per_second']:.2f} tok/s"
        ))

def interactive() -> None:
    console.print(Panel.fit(
        "[bold cyan]WeightLab Model Tester[/bold cyan]\n"
        "Compare an experiment against its original Hugging Face model",
        border_style="cyan",
    ))
    exp = choose_experiment()
    manifest = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
    repo_id = manifest["model"]["repo_id"]
    resolved = manifest["model"]["resolved_revision"]
    model_dir = exp / "model"

    prompt_file_text = Prompt.ask("Prompt file path (blank = built-ins)", default="")
    prompt_file = Path(prompt_file_text).expanduser() if prompt_file_text else None
    prompts = load_prompts(prompt_file)
    device = Prompt.ask("Device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    device = choose_device(device)
    max_new_tokens = IntPrompt.ask("Max new tokens", default=128)

    original_path = snapshot_download(repo_id=repo_id, revision=resolved)
    original = evaluate(original_path, prompts, device, max_new_tokens)
    transformed = evaluate(str(model_dir), prompts, device, max_new_tokens)
    print_comparison(original, transformed)

    report = {
        "experiment": str(exp),
        "manifest": manifest,
        "settings": {"device": device, "max_new_tokens": max_new_tokens},
        "original": original,
        "transformed": transformed,
    }
    out = exp / "evaluation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[green]Saved evaluation:[/green] {out}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive WeightLab model comparison")
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompts", type=Path)
    args = parser.parse_args()

    try:
        if args.experiment:
            exp = args.experiment.expanduser().resolve()
            manifest = json.loads((exp / "manifest.json").read_text(encoding="utf-8"))
            repo_id = manifest["model"]["repo_id"]
            resolved = manifest["model"]["resolved_revision"]
            original_path = snapshot_download(repo_id=repo_id, revision=resolved)
            prompts = load_prompts(args.prompts)
            device = choose_device(args.device)
            original = evaluate(original_path, prompts, device, args.max_new_tokens)
            transformed = evaluate(str(exp / "model"), prompts, device, args.max_new_tokens)
            print_comparison(original, transformed)
            report = {
                "experiment": str(exp),
                "manifest": manifest,
                "settings": {"device": device, "max_new_tokens": args.max_new_tokens},
                "original": original,
                "transformed": transformed,
            }
            (exp / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
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
