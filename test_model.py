#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
except ImportError:
    PromptSession = None
    KeyBindings = None


console = Console()


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

APP_HOME = Path(
    os.getenv(
        "WEIGHTLAB_HOME",
        Path.home() / ".weightlab",
    )
).expanduser()

EXPERIMENTS_DIR = APP_HOME / "experiments"


# Internal safety ceiling only.
#
# This is NOT the desired response length.
# The model should normally stop naturally using its own
# EOS / end-of-turn token from generation_config.json.
MAX_NEW_TOKENS = 2048


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

def choose_device(
    requested: str = "auto",
) -> str:

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
# Experiments
# ---------------------------------------------------------------------

def list_experiments() -> list[Path]:

    if not EXPERIMENTS_DIR.exists():
        return []

    experiments = [
        path
        for path in EXPERIMENTS_DIR.glob("*/*")
        if (
            path.is_dir()
            and (path / "manifest.json").exists()
        )
    ]

    return sorted(
        experiments,
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def read_manifest(
    experiment: Path,
) -> dict[str, Any]:

    manifest_path = (
        experiment
        / "manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found:\n"
            f"{manifest_path}"
        )

    return json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )


def choose_experiment() -> Path:

    experiments = list_experiments()

    if not experiments:
        raise FileNotFoundError(
            f"No WeightLab experiments found under:\n"
            f"{EXPERIMENTS_DIR}"
        )

    table = Table(
        title="WeightLab Experiments",
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

    for index, experiment in enumerate(
        experiments,
        start=1,
    ):

        try:
            manifest = read_manifest(
                experiment
            )

            model_name = (
                manifest
                .get("model", {})
                .get("repo_id", "?")
            )

            experiment_name = (
                manifest.get(
                    "experiment_name",
                    experiment.name,
                )
            )

            status = manifest.get(
                "status",
                "?",
            )

        except Exception:
            model_name = "?"
            experiment_name = (
                experiment.name
            )
            status = "invalid"

        table.add_row(
            str(index),
            model_name,
            experiment_name,
            status,
        )

    console.print()
    console.print(table)

    selection = IntPrompt.ask(
        "Choose experiment",
        default=1,
    )

    if not 1 <= selection <= len(
        experiments
    ):
        raise ValueError(
            "Invalid experiment selection."
        )

    return experiments[
        selection - 1
    ]


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

def choose_model_type() -> str:

    console.print()

    table = Table(
        show_header=False,
        box=None,
    )

    table.add_column(
        "#",
        style="cyan",
        width=3,
    )

    table.add_column(
        "Model",
    )

    table.add_row(
        "1",
        "Transformed model",
    )

    table.add_row(
        "2",
        "Original model",
    )

    table.add_row(
        "0",
        "Exit",
    )

    console.print(table)

    selection = IntPrompt.ask(
        "Choose model",
        default=1,
    )

    if selection == 0:
        return "exit"

    if selection == 1:
        return "transformed"

    if selection == 2:
        return "original"

    raise ValueError(
        "Invalid model selection."
    )


# ---------------------------------------------------------------------
# Resolve model
# ---------------------------------------------------------------------

def resolve_experiment_models(
    experiment: Path,
) -> tuple[
    dict[str, Any],
    Path,
    Path,
]:

    manifest = read_manifest(
        experiment
    )

    model_info = manifest.get(
        "model",
        {},
    )

    repo_id = model_info.get(
        "repo_id"
    )

    revision = model_info.get(
        "resolved_revision"
    )

    if not repo_id:
        raise ValueError(
            "Manifest does not contain "
            "model.repo_id"
        )

    if not revision:
        raise ValueError(
            "Manifest does not contain "
            "model.resolved_revision"
        )

    transformed_path = (
        experiment
        / "model"
    )

    if not transformed_path.exists():
        raise FileNotFoundError(
            "Transformed model directory "
            "does not exist:\n"
            f"{transformed_path}"
        )

    console.print()
    console.print(
        "[dim]"
        "Resolving exact original model revision..."
        "[/dim]"
    )

    original_path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
        )
    )

    return (
        manifest,
        original_path,
        transformed_path,
    )


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def load_chat_model(
    weights_path: Path,
    reference_path: Path,
    device: str,
):

    console.print()
    console.print(
        "[dim]"
        "Loading model..."
        "[/dim]"
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Always use the tokenizer from the exact ORIGINAL checkpoint.
    #
    # WeightLab experiments modify weights.
    # They should not modify tokenizer/chat formatting behavior.
    # -------------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(reference_path),
        )
    )

    # -------------------------------------------------------------
    # We intentionally require a real chat template.
    #
    # Do NOT invent:
    #
    #     User:
    #     Assistant:
    #
    # as a fallback.
    #
    # That was one source of the broken looping behavior.
    # -------------------------------------------------------------

    if not getattr(
        tokenizer,
        "chat_template",
        None,
    ):
        raise RuntimeError(
            "\nThis tokenizer does not define "
            "a chat template.\n\n"
            "The selected model may be a base "
            "language model rather than a "
            "chat/instruction model.\n\n"
            "WeightLab chat mode intentionally "
            "does not invent a User:/Assistant: "
            "prompt format."
        )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(weights_path),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )
    )

    # -------------------------------------------------------------
    # CRITICAL:
    #
    # Load generation configuration from the ORIGINAL model.
    #
    # Some chat models use:
    #
    #   eos_token_id = [EOS, END_OF_TURN]
    #
    # or another model-specific termination configuration.
    #
    # Replacing this with:
    #
    #   tokenizer.eos_token_id
    #
    # can remove the end-of-turn token and cause:
    #
    #   User: hi
    #   Assistant:
    #   User: hi
    #   Assistant:
    #   ...
    #
    # forever.
    #
    # So we preserve the exact original generation config.
    # -------------------------------------------------------------

    try:
        generation_config = (
            GenerationConfig
            .from_pretrained(
                str(reference_path)
            )
        )

        model.generation_config = (
            generation_config
        )

    except Exception as exc:

        # If the model repo doesn't have an explicit
        # generation_config.json, AutoModel's configuration
        # is still usable.
        console.print(
            "[dim]"
            "No separate generation_config.json "
            f"loaded ({exc}). "
            "Using the model's built-in "
            "generation configuration."
            "[/dim]"
        )

    model.eval()

    model.to(
        device
    )

    console.print(
        "[green]"
        "Model ready."
        "[/green]"
    )

    return (
        tokenizer,
        model,
    )


def unload_model(
    tokenizer,
    model,
) -> None:

    del model
    del tokenizer

    clear_memory()


# ---------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------

def create_chat_session():

    if (
        PromptSession is None
        or KeyBindings is None
    ):
        raise RuntimeError(
            "\nMulti-line chat requires "
            "'prompt_toolkit'.\n\n"
            "Install it with:\n\n"
            "    pip install prompt_toolkit\n"
        )

    bindings = KeyBindings()

    # -------------------------------------------------------------
    # ENTER
    # Send message.
    # -------------------------------------------------------------

    @bindings.add("enter")
    def send_message(
        event,
    ) -> None:

        event.current_buffer.validate_and_handle()

    # -------------------------------------------------------------
    # ALT+ENTER
    # Insert newline.
    #
    # Most terminals report Alt+Enter as:
    #
    #     Escape + Enter
    # -------------------------------------------------------------

    @bindings.add(
        "escape",
        "enter",
    )
    def insert_newline_alt(
        event,
    ) -> None:

        event.current_buffer.insert_text(
            "\n"
        )

    # -------------------------------------------------------------
    # CTRL+J
    # Alternative newline shortcut.
    # -------------------------------------------------------------

    @bindings.add("c-j")
    def insert_newline_ctrl_j(
        event,
    ) -> None:

        event.current_buffer.insert_text(
            "\n"
        )

    return PromptSession(
        multiline=True,
        key_bindings=bindings,
    )


def read_user_message(
    session,
) -> str | None:

    try:

        value = session.prompt(
            "You: ",
            multiline=True,
            prompt_continuation="   │ ",
        )

        return value

    except EOFError:
        return None

    except KeyboardInterrupt:
        console.print()
        return ""


# ---------------------------------------------------------------------
# Chat generation
# ---------------------------------------------------------------------

def build_chat_inputs(
    tokenizer,
    messages: list[
        dict[str, str]
    ],
    device: str,
) -> dict[str, torch.Tensor]:

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # tokenize=True means the chat template creates the
    # final token sequence directly.
    #
    # We do NOT:
    #
    #   apply_chat_template(tokenize=False)
    #       ↓
    #   tokenizer(formatted_text)
    #
    # because that can duplicate special tokens.
    # -------------------------------------------------------------

    inputs = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    )

    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


def generate_response(
    tokenizer,
    model,
    messages: list[
        dict[str, str]
    ],
    device: str,
) -> str:

    inputs = build_chat_inputs(
        tokenizer=tokenizer,
        messages=messages,
        device=device,
    )

    input_length = int(
        inputs[
            "input_ids"
        ].shape[-1]
    )

    # -------------------------------------------------------------
    # CRITICAL:
    #
    # DO NOT pass eos_token_id here.
    #
    # DO NOT pass pad_token_id here unless absolutely necessary.
    #
    # DO NOT replace the model's generation configuration.
    #
    # model.generate() will use model.generation_config,
    # which was loaded from the exact original checkpoint.
    #
    # MAX_NEW_TOKENS is only a runaway safety ceiling.
    # -------------------------------------------------------------

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    # -------------------------------------------------------------
    # Decode ONLY newly generated tokens.
    #
    # Never decode the prompt/history again.
    # -------------------------------------------------------------

    generated_tokens = output[
        0,
        input_length:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return response.strip()


# ---------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------

def chat(
    tokenizer,
    model,
    device: str,
    label: str,
) -> None:

    session = create_chat_session()

    messages: list[
        dict[str, str]
    ] = []

    console.print()
    console.print(
        f"[bold cyan]{label}[/bold cyan]"
    )

    console.print(
        "[dim]"
        "Enter = send   "
        "Alt+Enter / Ctrl+J = new line   "
        "/clear = clear chat   "
        "/exit = exit"
        "[/dim]"
    )

    console.print()

    while True:

        user_text = read_user_message(
            session
        )

        if user_text is None:
            break

        user_text = (
            user_text.rstrip()
        )

        if not user_text:
            continue

        command = (
            user_text
            .strip()
            .lower()
        )

        # ---------------------------------------------------------
        # Exit
        # ---------------------------------------------------------

        if command in {
            "/exit",
            "/quit",
        }:
            break

        # ---------------------------------------------------------
        # Clear conversation
        # ---------------------------------------------------------

        if command == "/clear":

            messages.clear()

            console.print(
                "[dim]"
                "Conversation cleared."
                "[/dim]"
            )

            console.print()

            continue

        # ---------------------------------------------------------
        # User message
        # ---------------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        try:

            response = (
                generate_response(
                    tokenizer=tokenizer,
                    model=model,
                    messages=messages,
                    device=device,
                )
            )

        except Exception:

            # Remove failed user turn so it doesn't
            # poison future conversation history.
            messages.pop()

            raise

        # ---------------------------------------------------------
        # Empty response
        # ---------------------------------------------------------

        if not response:

            # Don't fabricate an assistant response.
            #
            # Also remove the unanswered user message so
            # the next attempt starts from the previous
            # valid conversation state.
            messages.pop()

            console.print()
            console.print(
                "[yellow]"
                "Model returned an empty response."
                "[/yellow]"
            )

            console.print()

            continue

        # ---------------------------------------------------------
        # Store assistant response
        # ---------------------------------------------------------

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        # ---------------------------------------------------------
        # Normal chat output
        # ---------------------------------------------------------

        console.print()
        console.print(
            "[bold magenta]"
            "Model:"
            "[/bold magenta]"
        )

        console.print(
            response,
            markup=False,
        )

        console.print()


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------

def interactive(
    experiment_arg: Path | None = None,
    model_arg: str | None = None,
    device_arg: str = "auto",
) -> None:

    # -------------------------------------------------------------
    # Experiment
    # -------------------------------------------------------------

    if experiment_arg is None:

        experiment = (
            choose_experiment()
        )

    else:

        experiment = (
            experiment_arg
            .expanduser()
            .resolve()
        )

    (
        manifest,
        original_path,
        transformed_path,
    ) = resolve_experiment_models(
        experiment
    )

    repo_id = (
        manifest
        .get("model", {})
        .get(
            "repo_id",
            "unknown",
        )
    )

    experiment_name = (
        manifest.get(
            "experiment_name",
            experiment.name,
        )
    )

    # -------------------------------------------------------------
    # Model choice
    # -------------------------------------------------------------

    model_type = (
        model_arg
        if model_arg is not None
        else choose_model_type()
    )

    if model_type == "exit":
        return

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    if device_arg == "auto":

        selected_device = Prompt.ask(
            "Device",
            choices=[
                "auto",
                "mps",
                "cuda",
                "cpu",
            ],
            default="auto",
        )

    else:

        selected_device = (
            device_arg
        )

    device = choose_device(
        selected_device
    )

    # -------------------------------------------------------------
    # Select weights
    # -------------------------------------------------------------

    if model_type == "transformed":

        weights_path = (
            transformed_path
        )

        label = (
            "Transformed Model"
        )

    elif model_type == "original":

        weights_path = (
            original_path
        )

        label = (
            "Original Model"
        )

    else:

        raise ValueError(
            f"Unknown model type: "
            f"{model_type}"
        )

    console.print()
    console.print(
        f"[dim]"
        f"Model: {repo_id}\n"
        f"Experiment: {experiment_name}\n"
        f"Testing: {label}\n"
        f"Device: {device}"
        f"[/dim]"
    )

    # -------------------------------------------------------------
    # Load
    #
    # Notice that reference_path is ALWAYS original_path.
    #
    # Transformed:
    #
    #     weights       = transformed
    #     tokenizer     = original
    #     generation    = original
    #
    # Original:
    #
    #     weights       = original
    #     tokenizer     = original
    #     generation    = original
    #
    # This isolates the weight transformation.
    # -------------------------------------------------------------

    tokenizer = None
    model = None

    try:

        (
            tokenizer,
            model,
        ) = load_chat_model(
            weights_path=weights_path,
            reference_path=original_path,
            device=device,
        )

        chat(
            tokenizer=tokenizer,
            model=model,
            device=device,
            label=label,
        )

    finally:

        if model is not None:

            unload_model(
                tokenizer,
                model,
            )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Chat directly with a "
            "WeightLab original or "
            "transformed model."
        )
    )

    parser.add_argument(
        "--experiment",
        type=Path,
        help=(
            "WeightLab experiment directory. "
            "If omitted, choose interactively."
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "original",
            "transformed",
        ],
        help=(
            "Model to chat with. "
            "If omitted, choose interactively."
        ),
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "mps",
        ],
        default="auto",
    )

    args = parser.parse_args()

    try:

        interactive(
            experiment_arg=args.experiment,
            model_arg=args.model,
            device_arg=args.device,
        )

        return 0

    except KeyboardInterrupt:

        console.print(
            "\n[dim]Exited.[/dim]"
        )

        return 130

    except Exception:

        console.print_exception()

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )