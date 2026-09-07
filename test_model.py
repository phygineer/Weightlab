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
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

console = Console()

APP_HOME = Path(
    os.getenv(
        "WEIGHTLAB_HOME",
        Path.home() / ".weightlab",
    )
).expanduser()

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


# ---------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------

def choose_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()

    elif device == "mps":
        torch.mps.synchronize()


def clear_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------

def list_experiments() -> list[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []

    items = [
        path
        for path in EXPERIMENTS_DIR.glob("*/*")
        if (path / "manifest.json").exists()
    ]

    return sorted(
        items,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def choose_experiment() -> Path:
    experiments = list_experiments()

    if not experiments:
        raise FileNotFoundError(
            f"No experiments found under {EXPERIMENTS_DIR}"
        )

    table = Table(
        title="Saved WeightLab Experiments",
        show_lines=True,
    )

    table.add_column(
        "#",
        justify="right",
        style="cyan",
    )

    table.add_column(
        "Model",
        style="bold",
    )

    table.add_column(
        "Experiment",
    )

    table.add_column(
        "Status",
    )

    for i, path in enumerate(
        experiments,
        start=1,
    ):
        manifest = json.loads(
            (path / "manifest.json").read_text(
                encoding="utf-8"
            )
        )

        table.add_row(
            str(i),
            manifest
            .get("model", {})
            .get("repo_id", "?"),
            manifest.get(
                "experiment_name",
                path.name,
            ),
            manifest.get(
                "status",
                "?",
            ),
        )

    console.print(table)

    idx = IntPrompt.ask(
        "Choose experiment",
        default=1,
    )

    if not 1 <= idx <= len(experiments):
        raise ValueError(
            "Invalid selection"
        )

    return experiments[idx - 1]


# ---------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------

def load_prompts(
    path: Path | None,
) -> list[str]:

    if path is None:
        return DEFAULT_PROMPTS

    if path.suffix.lower() == ".json":
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if (
            not isinstance(data, list)
            or not all(
                isinstance(x, str)
                for x in data
            )
        ):
            raise ValueError(
                "Prompt JSON must be an array of strings"
            )

        return data

    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------

def format_prompt(
    tokenizer,
    prompt: str,
) -> str:

    if getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        return tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def format_chat_messages(
    tokenizer,
    messages: list[dict[str, str]],
) -> str:

    if getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    lines = []

    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]

        lines.append(
            f"{role}: {content}"
        )

    lines.append(
        "Assistant:"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_model(
    model_path: str,
    device: str,
):
    console.print(
        f"\n[cyan]Loading model:[/cyan] {model_path}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()
    model.to(device)

    return tokenizer, model


def unload_model(
    tokenizer,
    model,
) -> None:
    del model
    del tokenizer
    clear_memory()


# ---------------------------------------------------------------------
# Single generation
# ---------------------------------------------------------------------

def generate_once(
    tokenizer,
    model,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:

    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    input_tokens = int(
        encoded["input_ids"].shape[-1]
    )

    sync(device)

    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=(
                tokenizer.pad_token_id
                or tokenizer.eos_token_id
            ),
        )

    sync(device)

    elapsed = (
        time.perf_counter()
        - started
    )

    generated = output[
        0,
        input_tokens:
    ]

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    generated_tokens = int(
        generated.shape[-1]
    )

    return {
        "response": text,
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "generation_seconds": elapsed,
        "tokens_per_second": (
            generated_tokens / elapsed
            if elapsed > 0
            else 0.0
        ),
    }


# ---------------------------------------------------------------------
# Automated evaluation
# ---------------------------------------------------------------------

def evaluate(
    model_path: str,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:

    tokenizer, model = load_model(
        model_path,
        device,
    )

    rows = []

    for prompt in tqdm(
        prompts,
        desc="Generating",
        unit="prompt",
        dynamic_ncols=True,
    ):
        formatted = format_prompt(
            tokenizer,
            prompt,
        )

        result = generate_once(
            tokenizer,
            model,
            formatted,
            device,
            max_new_tokens,
        )

        rows.append(
            {
                "prompt": prompt,
                **result,
            }
        )

    unload_model(
        tokenizer,
        model,
    )

    return {
        "model": model_path,
        "device": device,
        "results": rows,
    }


def print_comparison(
    original: dict[str, Any],
    transformed: dict[str, Any],
) -> None:

    console.print(
        "\n[bold]Evaluation Comparison[/bold]"
    )

    for i, (
        original_row,
        transformed_row,
    ) in enumerate(
        zip(
            original["results"],
            transformed["results"],
        ),
        start=1,
    ):
        console.print(
            Panel.fit(
                f"[bold]Prompt {i}[/bold]\n"
                f"{original_row['prompt']}\n\n"
                f"[cyan]Original[/cyan]\n"
                f"{original_row['response']}\n\n"
                f"[magenta]Transformed[/magenta]\n"
                f"{transformed_row['response']}\n\n"
                f"[dim]"
                f"Original: "
                f"{original_row['tokens_per_second']:.2f} tok/s\n"
                f"Transformed: "
                f"{transformed_row['tokens_per_second']:.2f} tok/s"
                f"[/dim]"
            )
        )


# ---------------------------------------------------------------------
# Manual chat mode
# ---------------------------------------------------------------------

def chat_with_single_model(
    model_path: str,
    label: str,
    device: str,
    max_new_tokens: int,
) -> list[dict[str, str]]:

    tokenizer, model = load_model(
        model_path,
        device,
    )

    messages: list[
        dict[str, str]
    ] = []

    console.print(
        Panel.fit(
            f"[bold]{label} Chat[/bold]\n\n"
            "Commands:\n"
            "  /exit   Leave chat\n"
            "  /clear  Clear conversation history\n"
            "  /history Show conversation history",
            border_style="cyan",
        )
    )

    try:
        while True:
            user_text = Prompt.ask(
                "\n[bold green]You[/bold green]"
            )

            command = (
                user_text
                .strip()
                .lower()
            )

            if command in {
                "/exit",
                "/quit",
            }:
                break

            if command == "/clear":
                messages.clear()

                console.print(
                    "[yellow]"
                    "Conversation history cleared."
                    "[/yellow]"
                )

                continue

            if command == "/history":
                if not messages:
                    console.print(
                        "[dim]"
                        "No conversation history."
                        "[/dim]"
                    )
                else:
                    for message in messages:
                        role = (
                            message["role"]
                            .capitalize()
                        )

                        console.print(
                            f"[bold]{role}:[/bold] "
                            f"{message['content']}"
                        )

                continue

            messages.append(
                {
                    "role": "user",
                    "content": user_text,
                }
            )

            formatted = (
                format_chat_messages(
                    tokenizer,
                    messages,
                )
            )

            result = generate_once(
                tokenizer,
                model,
                formatted,
                device,
                max_new_tokens,
            )

            assistant_text = (
                result["response"]
                .strip()
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text,
                }
            )

            console.print(
                Panel(
                    assistant_text,
                    title=label,
                    border_style="magenta",
                )
            )

            console.print(
                "[dim]"
                f"{result['generated_tokens']} tokens | "
                f"{result['generation_seconds']:.2f}s | "
                f"{result['tokens_per_second']:.2f} tok/s"
                "[/dim]"
            )

    finally:
        unload_model(
            tokenizer,
            model,
        )

    return messages


def side_by_side_chat(
    original_path: str,
    transformed_path: str,
    device: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:

    console.print(
        Panel.fit(
            "[bold]Side-by-Side Chat[/bold]\n\n"
            "Each prompt is sent to both models.\n\n"
            "Commands:\n"
            "  /exit   Leave chat\n"
            "  /clear  Clear conversation history",
            border_style="cyan",
        )
    )

    original_tokenizer, original_model = (
        load_model(
            original_path,
            device,
        )
    )

    transformed_tokenizer = None
    transformed_model = None

    original_messages: list[
        dict[str, str]
    ] = []

    transformed_messages: list[
        dict[str, str]
    ] = []

    transcript: list[
        dict[str, Any]
    ] = []

    try:
        console.print(
            "\n[yellow]"
            "Loading both models simultaneously may require "
            "substantially more RAM/VRAM."
            "[/yellow]"
        )

        transformed_tokenizer, transformed_model = (
            load_model(
                transformed_path,
                device,
            )
        )

        while True:
            user_text = Prompt.ask(
                "\n[bold green]You[/bold green]"
            )

            command = (
                user_text
                .strip()
                .lower()
            )

            if command in {
                "/exit",
                "/quit",
            }:
                break

            if command == "/clear":
                original_messages.clear()
                transformed_messages.clear()

                console.print(
                    "[yellow]"
                    "Conversation history cleared."
                    "[/yellow]"
                )

                continue

            original_messages.append(
                {
                    "role": "user",
                    "content": user_text,
                }
            )

            transformed_messages.append(
                {
                    "role": "user",
                    "content": user_text,
                }
            )

            original_prompt = (
                format_chat_messages(
                    original_tokenizer,
                    original_messages,
                )
            )

            transformed_prompt = (
                format_chat_messages(
                    transformed_tokenizer,
                    transformed_messages,
                )
            )

            original_result = generate_once(
                original_tokenizer,
                original_model,
                original_prompt,
                device,
                max_new_tokens,
            )

            transformed_result = generate_once(
                transformed_tokenizer,
                transformed_model,
                transformed_prompt,
                device,
                max_new_tokens,
            )

            original_text = (
                original_result["response"]
                .strip()
            )

            transformed_text = (
                transformed_result["response"]
                .strip()
            )

            original_messages.append(
                {
                    "role": "assistant",
                    "content": original_text,
                }
            )

            transformed_messages.append(
                {
                    "role": "assistant",
                    "content": transformed_text,
                }
            )

            transcript.append(
                {
                    "user": user_text,
                    "original": original_result,
                    "transformed": transformed_result,
                }
            )

            console.print(
                Panel(
                    original_text,
                    title="Original",
                    border_style="cyan",
                )
            )

            console.print(
                "[dim]"
                f"{original_result['generated_tokens']} tokens | "
                f"{original_result['generation_seconds']:.2f}s | "
                f"{original_result['tokens_per_second']:.2f} tok/s"
                "[/dim]"
            )

            console.print(
                Panel(
                    transformed_text,
                    title="Transformed",
                    border_style="magenta",
                )
            )

            console.print(
                "[dim]"
                f"{transformed_result['generated_tokens']} tokens | "
                f"{transformed_result['generation_seconds']:.2f}s | "
                f"{transformed_result['tokens_per_second']:.2f} tok/s"
                "[/dim]"
            )

    finally:
        if transformed_model is not None:
            unload_model(
                transformed_tokenizer,
                transformed_model,
            )

        unload_model(
            original_tokenizer,
            original_model,
        )

    return transcript


def sequential_comparison_chat(
    original_path: str,
    transformed_path: str,
    device: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:

    console.print(
        Panel.fit(
            "[bold]Memory-Friendly Manual Comparison[/bold]\n\n"
            "For each prompt, the original model is tested first,\n"
            "then unloaded before the transformed model is tested.\n\n"
            "Conversation history is not maintained between turns.\n"
            "This mode uses much less memory.",
            border_style="cyan",
        )
    )

    transcript = []

    while True:
        user_text = Prompt.ask(
            "\n[bold green]Prompt[/bold green]"
        )

        command = (
            user_text
            .strip()
            .lower()
        )

        if command in {
            "/exit",
            "/quit",
        }:
            break

        original_tokenizer, original_model = (
            load_model(
                original_path,
                device,
            )
        )

        original_prompt = format_prompt(
            original_tokenizer,
            user_text,
        )

        original_result = generate_once(
            original_tokenizer,
            original_model,
            original_prompt,
            device,
            max_new_tokens,
        )

        unload_model(
            original_tokenizer,
            original_model,
        )

        transformed_tokenizer, transformed_model = (
            load_model(
                transformed_path,
                device,
            )
        )

        transformed_prompt = format_prompt(
            transformed_tokenizer,
            user_text,
        )

        transformed_result = generate_once(
            transformed_tokenizer,
            transformed_model,
            transformed_prompt,
            device,
            max_new_tokens,
        )

        unload_model(
            transformed_tokenizer,
            transformed_model,
        )

        transcript.append(
            {
                "user": user_text,
                "original": original_result,
                "transformed": transformed_result,
            }
        )

        console.print(
            Panel(
                original_result["response"],
                title="Original",
                border_style="cyan",
            )
        )

        console.print(
            Panel(
                transformed_result["response"],
                title="Transformed",
                border_style="magenta",
            )
        )

    return transcript


# ---------------------------------------------------------------------
# Evaluation menu
# ---------------------------------------------------------------------

def choose_test_mode() -> str:
    table = Table(
        title="Choose Evaluation Mode",
        show_lines=True,
    )

    table.add_column(
        "#",
        justify="right",
        style="cyan",
    )

    table.add_column(
        "Mode",
        style="bold",
    )

    table.add_column(
        "Description",
    )

    table.add_row(
        "1",
        "Automatic comparison",
        "Run the same fixed prompt suite against original and transformed models.",
    )

    table.add_row(
        "2",
        "Chat with transformed model",
        "Interactive multi-turn conversation with the transformed checkpoint.",
    )

    table.add_row(
        "3",
        "Chat with original model",
        "Interactive multi-turn conversation with the exact original model revision.",
    )

    table.add_row(
        "4",
        "Side-by-side chat",
        "Send every message to both models and compare responses live. Uses more RAM/VRAM.",
    )

    table.add_row(
        "5",
        "Memory-friendly manual comparison",
        "Send one prompt to each model sequentially. Lower memory use, but no multi-turn history.",
    )

    table.add_row(
        "0",
        "Exit",
        "Leave the tester.",
    )

    console.print(
        table
    )

    choice = IntPrompt.ask(
        "Selection",
        default=1,
    )

    mapping = {
        0: "exit",
        1: "automatic",
        2: "transformed_chat",
        3: "original_chat",
        4: "side_by_side",
        5: "sequential",
    }

    if choice not in mapping:
        raise ValueError(
            "Invalid test mode"
        )

    return mapping[choice]


# ---------------------------------------------------------------------
# Interactive app
# ---------------------------------------------------------------------

def interactive() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]"
            "WeightLab Model Tester"
            "[/bold cyan]\n"
            "Evaluate and manually chat with "
            "original vs transformed models",
            border_style="cyan",
        )
    )

    experiment = choose_experiment()

    manifest = json.loads(
        (
            experiment
            / "manifest.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    repo_id = (
        manifest["model"]
        ["repo_id"]
    )

    resolved_revision = (
        manifest["model"]
        ["resolved_revision"]
    )

    transformed_path = (
        experiment
        / "model"
    )

    console.print(
        f"\nModel: [cyan]{repo_id}[/cyan]"
    )

    console.print(
        f"Experiment: "
        f"[magenta]"
        f"{manifest.get('experiment_name', experiment.name)}"
        f"[/magenta]"
    )

    device = Prompt.ask(
        "Device",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "mps",
        ],
        default="auto",
    )

    device = choose_device(
        device
    )

    max_new_tokens = IntPrompt.ask(
        "Max new tokens",
        default=128,
    )

    original_path = snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
    )

    while True:
        mode = choose_test_mode()

        if mode == "exit":
            break

        if mode == "automatic":
            prompt_file_text = Prompt.ask(
                "Prompt file path "
                "(blank = built-ins)",
                default="",
            )

            prompt_file = (
                Path(
                    prompt_file_text
                ).expanduser()
                if prompt_file_text
                else None
            )

            prompts = load_prompts(
                prompt_file
            )

            original = evaluate(
                original_path,
                prompts,
                device,
                max_new_tokens,
            )

            transformed = evaluate(
                str(transformed_path),
                prompts,
                device,
                max_new_tokens,
            )

            print_comparison(
                original,
                transformed,
            )

            report = {
                "experiment": str(
                    experiment
                ),
                "manifest": manifest,
                "settings": {
                    "device": device,
                    "max_new_tokens": (
                        max_new_tokens
                    ),
                    "mode": "automatic",
                },
                "original": original,
                "transformed": transformed,
            }

            output = (
                experiment
                / "evaluation.json"
            )

            output.write_text(
                json.dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            console.print(
                f"\n[green]"
                f"Saved evaluation:"
                f"[/green] {output}"
            )

        elif mode == "transformed_chat":
            history = chat_with_single_model(
                str(
                    transformed_path
                ),
                "Transformed Model",
                device,
                max_new_tokens,
            )

            output = (
                experiment
                / "manual_chat_transformed.json"
            )

            output.write_text(
                json.dumps(
                    history,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            console.print(
                f"[green]"
                f"Saved chat:"
                f"[/green] {output}"
            )

        elif mode == "original_chat":
            history = chat_with_single_model(
                original_path,
                "Original Model",
                device,
                max_new_tokens,
            )

            output = (
                experiment
                / "manual_chat_original.json"
            )

            output.write_text(
                json.dumps(
                    history,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            console.print(
                f"[green]"
                f"Saved chat:"
                f"[/green] {output}"
            )

        elif mode == "side_by_side":
            transcript = side_by_side_chat(
                original_path,
                str(
                    transformed_path
                ),
                device,
                max_new_tokens,
            )

            output = (
                experiment
                / "manual_chat_comparison.json"
            )

            output.write_text(
                json.dumps(
                    transcript,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            console.print(
                f"[green]"
                f"Saved comparison:"
                f"[/green] {output}"
            )

        elif mode == "sequential":
            transcript = (
                sequential_comparison_chat(
                    original_path,
                    str(
                        transformed_path
                    ),
                    device,
                    max_new_tokens,
                )
            )

            output = (
                experiment
                / "manual_prompt_comparison.json"
            )

            output.write_text(
                json.dumps(
                    transcript,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            console.print(
                f"[green]"
                f"Saved comparison:"
                f"[/green] {output}"
            )

        if not Confirm.ask(
            "Return to evaluation menu?",
            default=True,
        ):
            break


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WeightLab model evaluation "
            "and interactive chat tester"
        )
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        help=(
            "Optional experiment path. "
            "Without this, interactive mode is used."
        ),
    )

    args = parser.parse_args()

    try:
        if args.experiment:
            console.print(
                "[yellow]"
                "Direct --experiment mode currently "
                "starts the interactive evaluation menu."
                "[/yellow]"
            )

        interactive()

        return 0

    except KeyboardInterrupt:
        console.print(
            "\n[yellow]"
            "Cancelled."
            "[/yellow]"
        )

        return 130

    except Exception:
        console.print_exception()

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )