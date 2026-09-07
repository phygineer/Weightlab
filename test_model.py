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
                "Prompt JSON must contain an array of strings"
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
# Generation helpers
# ---------------------------------------------------------------------

def get_pad_token_id(tokenizer) -> int | None:
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id

    return tokenizer.eos_token_id


def generate_prompt_response(
    tokenizer,
    model,
    prompt: str,
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:

    if getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        inputs = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

    input_length = int(
        inputs["input_ids"].shape[-1]
    )

    sync(device)

    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=get_pad_token_id(
                tokenizer
            ),
        )

    sync(device)

    elapsed = (
        time.perf_counter()
        - started
    )

    generated = output[
        0,
        input_length:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    raw_response = tokenizer.decode(
        generated,
        skip_special_tokens=False,
    )

    token_ids = (
        generated
        .detach()
        .cpu()
        .tolist()
    )

    return {
        "response": response,
        "raw_response": raw_response,
        "generated_token_ids": token_ids,
        "input_tokens": input_length,
        "generated_tokens": len(token_ids),
        "generation_seconds": elapsed,
        "tokens_per_second": (
            len(token_ids) / elapsed
            if elapsed > 0
            else 0.0
        ),
    }


def generate_chat_response(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    device: str,
    max_new_tokens: int = 512,
) -> dict[str, Any]:

    if getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

    else:
        text_parts = []

        for message in messages:
            role = (
                "User"
                if message["role"] == "user"
                else "Assistant"
            )

            text_parts.append(
                f"{role}: {message['content']}"
            )

        text_parts.append(
            "Assistant:"
        )

        text = "\n".join(
            text_parts
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

    input_length = int(
        inputs["input_ids"].shape[-1]
    )

    sync(device)

    started = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(
            **inputs,

            # Safety ceiling only.
            # Normal chat models should stop naturally
            # when they emit EOS/end-of-turn.
            max_new_tokens=max_new_tokens,

            do_sample=False,
            use_cache=True,

            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=get_pad_token_id(
                tokenizer
            ),
        )

    sync(device)

    elapsed = (
        time.perf_counter()
        - started
    )

    generated = output[
        0,
        input_length:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()

    raw_response = tokenizer.decode(
        generated,
        skip_special_tokens=False,
    )

    token_ids = (
        generated
        .detach()
        .cpu()
        .tolist()
    )

    return {
        "response": response,
        "raw_response": raw_response,
        "generated_token_ids": token_ids,
        "generated_tokens": len(token_ids),
        "generation_seconds": elapsed,
        "tokens_per_second": (
            len(token_ids) / elapsed
            if elapsed > 0
            else 0.0
        ),
    }


# ---------------------------------------------------------------------
# Automatic evaluation
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

    results = []

    try:
        for prompt in tqdm(
            prompts,
            desc="Generating",
            unit="prompt",
            dynamic_ncols=True,
        ):

            result = generate_prompt_response(
                tokenizer,
                model,
                prompt,
                device,
                max_new_tokens,
            )

            results.append(
                {
                    "prompt": prompt,
                    **result,
                }
            )

    finally:
        unload_model(
            tokenizer,
            model,
        )

    return {
        "model": model_path,
        "device": device,
        "results": results,
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

        original_text = (
            original_row["response"]
            or (
                "[NO VISIBLE OUTPUT]\n"
                f"Raw: "
                f"{original_row['raw_response']!r}"
            )
        )

        transformed_text = (
            transformed_row["response"]
            or (
                "[NO VISIBLE OUTPUT]\n"
                f"Raw: "
                f"{transformed_row['raw_response']!r}"
            )
        )

        console.print(
            Panel(
                (
                    f"[bold]Prompt[/bold]\n"
                    f"{original_row['prompt']}\n\n"

                    f"[cyan]Original[/cyan]\n"
                    f"{original_text}\n\n"

                    f"[magenta]Transformed[/magenta]\n"
                    f"{transformed_text}"
                ),
                title=f"Prompt {i}",
            )
        )


# ---------------------------------------------------------------------
# Normal single-model chat
# ---------------------------------------------------------------------

def chat_with_single_model(
    model_path: str,
    label: str,
    device: str,
) -> list[dict[str, Any]]:

    tokenizer, model = load_model(
        model_path,
        device,
    )

    messages: list[
        dict[str, str]
    ] = []

    transcript: list[
        dict[str, Any]
    ] = []

    console.print(
        Panel.fit(
            f"[bold]{label}[/bold]\n\n"
            "Chat normally.\n\n"
            "Commands:\n"
            "  /clear   Clear conversation\n"
            "  /history Show conversation\n"
            "  /exit    Exit chat",
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
                    "[dim]"
                    "Conversation cleared."
                    "[/dim]"
                )

                continue

            if command == "/history":
                if not messages:
                    console.print(
                        "[dim]"
                        "No conversation history."
                        "[/dim]"
                    )

                    continue

                console.print()

                for message in messages:
                    if message["role"] == "user":
                        console.print(
                            f"[bold green]"
                            f"You:"
                            f"[/bold green] "
                            f"{message['content']}"
                        )

                    else:
                        console.print(
                            f"[bold magenta]"
                            f"{label}:"
                            f"[/bold magenta] "
                            f"{message['content']}"
                        )

                continue

            messages.append(
                {
                    "role": "user",
                    "content": user_text,
                }
            )

            result = (
                generate_chat_response(
                    tokenizer=tokenizer,
                    model=model,
                    messages=messages,
                    device=device,
                )
            )

            response = (
                result["response"]
                .strip()
            )

            if not response:
                console.print(
                    Panel(
                        (
                            "[yellow]"
                            "The model produced no visible response."
                            "[/yellow]\n\n"
                            f"Raw output: "
                            f"{result['raw_response']!r}\n"
                            f"Token IDs: "
                            f"{result['generated_token_ids']}"
                        ),
                        title=label,
                        border_style="yellow",
                    )
                )

                # Remove the unanswered user message so that
                # repeated failed turns don't accumulate.
                messages.pop()

                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            transcript.append(
                {
                    "user": user_text,
                    "assistant": response,
                    "generated_tokens": (
                        result["generated_tokens"]
                    ),
                    "generation_seconds": (
                        result[
                            "generation_seconds"
                        ]
                    ),
                    "tokens_per_second": (
                        result[
                            "tokens_per_second"
                        ]
                    ),
                }
            )

            console.print(
                Panel(
                    response,
                    title=label,
                    border_style="magenta",
                )
            )

    finally:
        unload_model(
            tokenizer,
            model,
        )

    return transcript


# ---------------------------------------------------------------------
# Side-by-side normal chat
# ---------------------------------------------------------------------

def side_by_side_chat(
    original_path: str,
    transformed_path: str,
    device: str,
) -> list[dict[str, Any]]:

    console.print(
        Panel.fit(
            "[bold]Side-by-Side Chat[/bold]\n\n"
            "Every message is sent to both models.\n"
            "Both models maintain their own conversation history.\n\n"
            "Commands:\n"
            "  /clear   Clear both conversations\n"
            "  /exit    Exit chat",
            border_style="cyan",
        )
    )

    console.print(
        "[yellow]"
        "This mode loads both models at once "
        "and therefore uses more RAM/VRAM."
        "[/yellow]"
    )

    original_tokenizer = None
    original_model = None

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
        (
            original_tokenizer,
            original_model,
        ) = load_model(
            original_path,
            device,
        )

        (
            transformed_tokenizer,
            transformed_model,
        ) = load_model(
            transformed_path,
            device,
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
                    "[dim]"
                    "Conversation histories cleared."
                    "[/dim]"
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

            original_result = (
                generate_chat_response(
                    tokenizer=original_tokenizer,
                    model=original_model,
                    messages=original_messages,
                    device=device,
                )
            )

            transformed_result = (
                generate_chat_response(
                    tokenizer=transformed_tokenizer,
                    model=transformed_model,
                    messages=transformed_messages,
                    device=device,
                )
            )

            original_response = (
                original_result[
                    "response"
                ].strip()
            )

            transformed_response = (
                transformed_result[
                    "response"
                ].strip()
            )

            if original_response:
                original_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            original_response
                        ),
                    }
                )

            else:
                original_messages.pop()

            if transformed_response:
                transformed_messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            transformed_response
                        ),
                    }
                )

            else:
                transformed_messages.pop()

            transcript.append(
                {
                    "user": user_text,
                    "original": (
                        original_result
                    ),
                    "transformed": (
                        transformed_result
                    ),
                }
            )

            if original_response:
                console.print(
                    Panel(
                        original_response,
                        title="Original Model",
                        border_style="cyan",
                    )
                )

            else:
                console.print(
                    Panel(
                        (
                            "[yellow]"
                            "No visible response.\n\n"
                            f"Raw: "
                            f"{original_result['raw_response']!r}"
                        ),
                        title="Original Model",
                        border_style="yellow",
                    )
                )

            if transformed_response:
                console.print(
                    Panel(
                        transformed_response,
                        title=(
                            "Transformed Model"
                        ),
                        border_style="magenta",
                    )
                )

            else:
                console.print(
                    Panel(
                        (
                            "[yellow]"
                            "No visible response.\n\n"
                            f"Raw: "
                            f"{transformed_result['raw_response']!r}"
                        ),
                        title=(
                            "Transformed Model"
                        ),
                        border_style="yellow",
                    )
                )

    finally:

        if transformed_model is not None:
            unload_model(
                transformed_tokenizer,
                transformed_model,
            )

        if original_model is not None:
            unload_model(
                original_tokenizer,
                original_model,
            )

    return transcript


# ---------------------------------------------------------------------
# Memory-friendly one-prompt comparison
# ---------------------------------------------------------------------

def sequential_comparison_chat(
    original_path: str,
    transformed_path: str,
    device: str,
) -> list[dict[str, Any]]:

    console.print(
        Panel.fit(
            "[bold]"
            "Memory-Friendly Manual Comparison"
            "[/bold]\n\n"
            "Enter a prompt and WeightLab runs it "
            "against each model sequentially.\n\n"
            "This mode does not preserve multi-turn history.\n\n"
            "Command:\n"
            "  /exit    Exit",
            border_style="cyan",
        )
    )

    transcript = []

    while True:

        user_text = Prompt.ask(
            "\n[bold green]You[/bold green]"
        )

        if (
            user_text
            .strip()
            .lower()
            in {
                "/exit",
                "/quit",
            }
        ):
            break

        # ---------------------------------------------------------
        # Original
        # ---------------------------------------------------------

        (
            original_tokenizer,
            original_model,
        ) = load_model(
            original_path,
            device,
        )

        try:
            original_result = (
                generate_prompt_response(
                    tokenizer=(
                        original_tokenizer
                    ),
                    model=original_model,
                    prompt=user_text,
                    device=device,
                    max_new_tokens=512,
                )
            )

        finally:
            unload_model(
                original_tokenizer,
                original_model,
            )

        # ---------------------------------------------------------
        # Transformed
        # ---------------------------------------------------------

        (
            transformed_tokenizer,
            transformed_model,
        ) = load_model(
            transformed_path,
            device,
        )

        try:
            transformed_result = (
                generate_prompt_response(
                    tokenizer=(
                        transformed_tokenizer
                    ),
                    model=(
                        transformed_model
                    ),
                    prompt=user_text,
                    device=device,
                    max_new_tokens=512,
                )
            )

        finally:
            unload_model(
                transformed_tokenizer,
                transformed_model,
            )

        transcript.append(
            {
                "user": user_text,
                "original": original_result,
                "transformed": (
                    transformed_result
                ),
            }
        )

        console.print(
            Panel(
                (
                    original_result[
                        "response"
                    ]
                    or (
                        "[NO VISIBLE RESPONSE]\n"
                        f"{original_result['raw_response']!r}"
                    )
                ),
                title="Original Model",
                border_style="cyan",
            )
        )

        console.print(
            Panel(
                (
                    transformed_result[
                        "response"
                    ]
                    or (
                        "[NO VISIBLE RESPONSE]\n"
                        f"{transformed_result['raw_response']!r}"
                    )
                ),
                title="Transformed Model",
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
        (
            "Run a fixed prompt suite against "
            "the original and transformed models."
        ),
    )

    table.add_row(
        "2",
        "Chat with transformed model",
        (
            "Normal multi-turn chat using "
            "the transformed checkpoint."
        ),
    )

    table.add_row(
        "3",
        "Chat with original model",
        (
            "Normal multi-turn chat using "
            "the exact original model."
        ),
    )

    table.add_row(
        "4",
        "Side-by-side chat",
        (
            "Chat with both models at the same time "
            "and compare their replies."
        ),
    )

    table.add_row(
        "5",
        "Memory-friendly comparison",
        (
            "Enter one prompt at a time. "
            "Models are loaded sequentially."
        ),
    )

    table.add_row(
        "0",
        "Exit",
        "Leave the tester.",
    )

    console.print(table)

    choice = IntPrompt.ask(
        "Selection",
        default=2,
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
# Experiment loading
# ---------------------------------------------------------------------

def load_experiment(
    experiment: Path,
) -> tuple[
    dict[str, Any],
    str,
    Path,
]:

    manifest_path = (
        experiment
        / "manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found: "
            f"{manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    transformed_path = (
        experiment
        / "model"
    )

    if not transformed_path.exists():
        raise FileNotFoundError(
            f"Transformed model directory "
            f"not found: {transformed_path}"
        )

    repo_id = (
        manifest["model"]
        ["repo_id"]
    )

    resolved_revision = (
        manifest["model"]
        ["resolved_revision"]
    )

    console.print(
        f"\nResolving original model "
        f"[cyan]{repo_id}[/cyan]..."
    )

    original_path = snapshot_download(
        repo_id=repo_id,
        revision=resolved_revision,
    )

    return (
        manifest,
        original_path,
        transformed_path,
    )


# ---------------------------------------------------------------------
# Main interactive app
# ---------------------------------------------------------------------

def interactive(
    selected_experiment: Path | None = None,
) -> None:

    console.print(
        Panel.fit(
            "[bold cyan]"
            "WeightLab Model Tester"
            "[/bold cyan]\n"
            "Automatic evaluation and normal interactive chat",
            border_style="cyan",
        )
    )

    experiment = (
        selected_experiment
        if selected_experiment is not None
        else choose_experiment()
    )

    experiment = (
        experiment
        .expanduser()
        .resolve()
    )

    (
        manifest,
        original_path,
        transformed_path,
    ) = load_experiment(
        experiment
    )

    console.print(
        f"\nModel: "
        f"[cyan]"
        f"{manifest['model']['repo_id']}"
        f"[/cyan]"
    )

    console.print(
        f"Experiment: "
        f"[magenta]"
        f"{manifest.get('experiment_name', experiment.name)}"
        f"[/magenta]"
    )

    console.print(
        f"Revision: "
        f"[dim]"
        f"{manifest['model']['resolved_revision']}"
        f"[/dim]"
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

    console.print(
        f"Using device: "
        f"[cyan]{device}[/cyan]"
    )

    while True:

        mode = choose_test_mode()

        if mode == "exit":
            break

        # ---------------------------------------------------------
        # Automatic evaluation
        # ---------------------------------------------------------

        if mode == "automatic":

            prompt_file_text = Prompt.ask(
                "Prompt file "
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

            max_new_tokens = (
                IntPrompt.ask(
                    "Maximum generation tokens",
                    default=128,
                )
            )

            original = evaluate(
                original_path,
                prompts,
                device,
                max_new_tokens,
            )

            transformed = evaluate(
                str(
                    transformed_path
                ),
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
                "mode": "automatic",
                "device": device,
                "max_new_tokens": (
                    max_new_tokens
                ),
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

        # ---------------------------------------------------------
        # Transformed chat
        # ---------------------------------------------------------

        elif mode == "transformed_chat":

            transcript = (
                chat_with_single_model(
                    str(
                        transformed_path
                    ),
                    "Transformed Model",
                    device,
                )
            )

            output = (
                experiment
                / "manual_chat_transformed.json"
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
                f"\n[green]"
                f"Saved chat:"
                f"[/green] {output}"
            )

        # ---------------------------------------------------------
        # Original chat
        # ---------------------------------------------------------

        elif mode == "original_chat":

            transcript = (
                chat_with_single_model(
                    original_path,
                    "Original Model",
                    device,
                )
            )

            output = (
                experiment
                / "manual_chat_original.json"
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
                f"\n[green]"
                f"Saved chat:"
                f"[/green] {output}"
            )

        # ---------------------------------------------------------
        # Side-by-side
        # ---------------------------------------------------------

        elif mode == "side_by_side":

            transcript = (
                side_by_side_chat(
                    original_path,
                    str(
                        transformed_path
                    ),
                    device,
                )
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
                f"\n[green]"
                f"Saved comparison:"
                f"[/green] {output}"
            )

        # ---------------------------------------------------------
        # Sequential comparison
        # ---------------------------------------------------------

        elif mode == "sequential":

            transcript = (
                sequential_comparison_chat(
                    original_path,
                    str(
                        transformed_path
                    ),
                    device,
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
                f"\n[green]"
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
            "WeightLab model evaluator "
            "and interactive chat tester"
        )
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        help=(
            "Optional WeightLab experiment directory. "
            "If omitted, choose interactively."
        ),
    )

    args = parser.parse_args()

    try:

        interactive(
            selected_experiment=(
                args.experiment
                if args.experiment
                else None
            )
        )

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