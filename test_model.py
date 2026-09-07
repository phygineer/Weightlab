#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

console = Console()

APP_HOME = Path(
    os.getenv("WEIGHTLAB_HOME", Path.home() / ".weightlab")
).expanduser()

EXPERIMENTS_DIR = APP_HOME / "experiments"

# Runaway protection only.
# The model should normally stop much earlier using EOS/end-of-turn.
MAX_NEW_TOKENS = 2048


# -----------------------------------------------------------------------------
# Device helpers
# -----------------------------------------------------------------------------

def choose_device(requested: str) -> str:
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


# -----------------------------------------------------------------------------
# Experiment discovery
# -----------------------------------------------------------------------------

def read_manifest(
    experiment: Path,
) -> dict[str, Any]:

    path = experiment / "manifest.json"

    if not path.exists():
        raise FileNotFoundError(
            f"manifest.json not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


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


def choose_experiment() -> Path:
    experiments = list_experiments()

    if not experiments:
        raise FileNotFoundError(
            f"No WeightLab experiments found under "
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
            experiment_name = experiment.name
            status = "invalid"

        table.add_row(
            str(index),
            model_name,
            experiment_name,
            status,
        )

    console.print()
    console.print(table)

    choice = IntPrompt.ask(
        "Choose experiment",
        default=1,
    )

    if not 1 <= choice <= len(
        experiments
    ):
        raise ValueError(
            "Invalid experiment selection"
        )

    return experiments[
        choice - 1
    ]


def choose_model_type() -> str:
    table = Table(
        show_header=False,
        box=None,
    )

    table.add_column(
        "#",
        width=3,
        style="cyan",
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

    console.print()
    console.print(table)

    choice = IntPrompt.ask(
        "Choose model",
        default=1,
    )

    if choice == 0:
        return "exit"

    if choice == 1:
        return "transformed"

    if choice == 2:
        return "original"

    raise ValueError(
        "Invalid model selection"
    )


def resolve_models(
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
            "Manifest is missing model.repo_id"
        )

    if not revision:
        raise ValueError(
            "Manifest is missing "
            "model.resolved_revision"
        )

    transformed_path = (
        experiment / "model"
    )

    if not transformed_path.exists():
        raise FileNotFoundError(
            f"Transformed model directory "
            f"not found: {transformed_path}"
        )

    console.print(
        "\n[dim]"
        "Resolving exact original checkpoint..."
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


# -----------------------------------------------------------------------------
# Model/tokenizer loading
# -----------------------------------------------------------------------------

def load_reference_generation_config(
    reference_path: Path,
    model,
    tokenizer,
) -> GenerationConfig:
    """
    Use the exact original model's generation config
    when available.

    Only max_new_tokens is modified.

    We do NOT pass max_new_tokens separately to
    model.generate(), avoiding the Transformers
    generation_config + generation kwargs warning.
    """

    try:
        config = (
            GenerationConfig
            .from_pretrained(
                str(reference_path),
                local_files_only=True,
            )
        )

    except Exception:
        config = deepcopy(
            model.generation_config
        )

    config = deepcopy(
        config
    )

    # Safety ceiling, not target response length.
    config.max_new_tokens = (
        MAX_NEW_TOKENS
    )

    # Decoder-only models sometimes do not
    # explicitly configure padding.
    if config.pad_token_id is None:

        if tokenizer.pad_token_id is not None:
            config.pad_token_id = (
                tokenizer.pad_token_id
            )

        elif tokenizer.eos_token_id is not None:
            config.pad_token_id = (
                tokenizer.eos_token_id
            )

    return config


def tokenizer_has_working_chat_template(
    tokenizer,
) -> bool:
    """
    Instead of only checking tokenizer.chat_template,
    actually test apply_chat_template().

    This also works with tokenizers exposing named
    templates/default templates.
    """

    try:
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "test",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        return True

    except Exception:
        return False


def load_chat_model(
    weights_path: Path,
    reference_path: Path,
    device: str,
):
    console.print(
        "\n[dim]Loading model...[/dim]"
    )

    # Always use tokenizer/chat metadata from
    # the untouched ORIGINAL checkpoint.
    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(reference_path),
            local_files_only=True,
        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(weights_path),
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    )

    model.eval()

    model.to(
        device
    )

    generation_config = (
        load_reference_generation_config(
            reference_path=reference_path,
            model=model,
            tokenizer=tokenizer,
        )
    )

    native_chat = (
        tokenizer_has_working_chat_template(
            tokenizer
        )
    )

    console.print(
        "[green]Model ready.[/green]"
    )

    if native_chat:
        console.print(
            "[dim]"
            "Chat format: native model template"
            "[/dim]"
        )
    else:
        console.print(
            "[dim]"
            "Chat format: plain fallback"
            "[/dim]"
        )

    return (
        tokenizer,
        model,
        generation_config,
        native_chat,
    )


def unload_model(
    tokenizer,
    model,
) -> None:

    del model
    del tokenizer

    clear_memory()


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

def build_native_inputs(
    tokenizer,
    messages: list[
        dict[str, str]
    ],
    device: str,
) -> dict[
    str,
    torch.Tensor,
]:
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


def render_plain_chat(
    messages: list[
        dict[str, str]
    ],
) -> str:
    """
    Fallback for checkpoints that do not provide
    a chat template.

    This is intentionally simple:

        User: hello
        Assistant: Hi!
        User: ...
        Assistant:
    """

    parts: list[str] = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            parts.append(
                f"User: {content}"
            )

        elif role == "assistant":
            parts.append(
                f"Assistant: {content}"
            )

        elif role == "system":
            parts.append(
                f"System: {content}"
            )

    parts.append(
        "Assistant:"
    )

    return "\n".join(
        parts
    )


def build_plain_inputs(
    tokenizer,
    messages: list[
        dict[str, str]
    ],
    device: str,
) -> dict[
    str,
    torch.Tensor,
]:

    prompt = render_plain_chat(
        messages
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


# -----------------------------------------------------------------------------
# Response cleanup
# -----------------------------------------------------------------------------

def trim_fallback_role_continuation(
    text: str,
) -> str:
    """
    A model without a real chat template can sometimes
    continue by generating another fake User:/Assistant:
    turn.

    Keep only the first assistant response.
    """

    markers = (
        "\nUser:",
        "\nHuman:",
        "\nAssistant:",
        "\nSystem:",
        "\n### User:",
        "\n### Human:",
        "\n### Assistant:",
        "\n### System:",
    )

    cut = len(
        text
    )

    for marker in markers:
        index = text.find(
            marker
        )

        if index != -1:
            cut = min(
                cut,
                index,
            )

    return text[
        :cut
    ].strip()


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

def generate_response(
    tokenizer,
    model,
    generation_config: GenerationConfig,
    messages: list[
        dict[str, str]
    ],
    device: str,
    native_chat: bool,
) -> str:

    # -----------------------------------------------------------------
    # Construct prompt
    # -----------------------------------------------------------------

    if native_chat:

        inputs = build_native_inputs(
            tokenizer=tokenizer,
            messages=messages,
            device=device,
        )

    else:

        inputs = build_plain_inputs(
            tokenizer=tokenizer,
            messages=messages,
            device=device,
        )

    prompt_length = int(
        inputs[
            "input_ids"
        ].shape[-1]
    )

    # -----------------------------------------------------------------
    # Per-turn generation config
    #
    # IMPORTANT:
    #
    # We pass ONE generation_config.
    #
    # We do NOT also pass:
    #
    #     max_new_tokens=...
    #     eos_token_id=...
    #     pad_token_id=...
    #
    # This avoids the warning you were seeing and
    # preserves the original model's EOS/EOT settings.
    # -----------------------------------------------------------------

    config = deepcopy(
        generation_config
    )

    # -----------------------------------------------------------------
    # Model without chat template
    #
    # Stop it before it starts generating:
    #
    #     User:
    #     Human:
    #
    # as another fake turn.
    # -----------------------------------------------------------------

    if not native_chat:

        config.stop_strings = [
            "\nUser:",
            "\nHuman:",
            "\n### User:",
            "\n### Human:",
        ]

    # -----------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------

    with torch.inference_mode():

        if native_chat:

            output = model.generate(
                **inputs,
                generation_config=config,
            )

        else:

            # tokenizer= is required by Transformers
            # when stop_strings are configured.
            output = model.generate(
                **inputs,
                generation_config=config,
                tokenizer=tokenizer,
            )

    # -----------------------------------------------------------------
    # Decoder-only CausalLM output contains:
    #
    #     [prompt tokens][new tokens]
    #
    # Decode ONLY the assistant's new tokens.
    # -----------------------------------------------------------------

    generated = output[
        0,
        prompt_length:
    ]

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    if not native_chat:

        response = (
            trim_fallback_role_continuation(
                response
            )
        )

    return response


# -----------------------------------------------------------------------------
# Multi-line terminal input
# -----------------------------------------------------------------------------

def create_prompt_session() -> PromptSession:

    bindings = KeyBindings()

    # -------------------------------------------------------------
    # Enter
    #
    # Send the current message.
    # -------------------------------------------------------------

    @bindings.add(
        "enter"
    )
    def send_message(
        event,
    ) -> None:

        event.current_buffer.validate_and_handle()

    # -------------------------------------------------------------
    # Alt+Enter
    #
    # Insert newline.
    #
    # Many terminals represent Alt+Enter as
    # Escape followed by Enter.
    # -------------------------------------------------------------

    @bindings.add(
        "escape",
        "enter",
    )
    def alt_enter(
        event,
    ) -> None:

        event.current_buffer.insert_text(
            "\n"
        )

    # -------------------------------------------------------------
    # Ctrl+J
    #
    # Alternative newline shortcut.
    # -------------------------------------------------------------

    @bindings.add(
        "c-j"
    )
    def ctrl_j(
        event,
    ) -> None:

        event.current_buffer.insert_text(
            "\n"
        )

    return PromptSession(
        multiline=True,
        key_bindings=bindings,
    )


# -----------------------------------------------------------------------------
# Normal chat loop
# -----------------------------------------------------------------------------

def chat(
    tokenizer,
    model,
    generation_config: GenerationConfig,
    native_chat: bool,
    device: str,
    label: str,
) -> None:

    session = create_prompt_session()

    messages: list[
        dict[str, str]
    ] = []

    console.print()

    console.print(
        f"[bold cyan]"
        f"{label}"
        f"[/bold cyan]"
    )

    console.print(
        "[dim]"
        "Enter = send   "
        "Alt+Enter/Ctrl+J = newline   "
        "/clear = clear   "
        "/exit = exit"
        "[/dim]"
    )

    console.print()

    while True:

        # ---------------------------------------------------------
        # Read user message
        # ---------------------------------------------------------

        try:

            user_text = session.prompt(
                "You: ",
                multiline=True,
                prompt_continuation=(
                    "   │ "
                ),
            )

        except EOFError:
            break

        except KeyboardInterrupt:
            console.print()
            continue

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
                "[/dim]\n"
            )

            continue

        # ---------------------------------------------------------
        # Add user turn
        # ---------------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        # ---------------------------------------------------------
        # Generate
        # ---------------------------------------------------------

        try:

            response = (
                generate_response(
                    tokenizer=tokenizer,
                    model=model,
                    generation_config=(
                        generation_config
                    ),
                    messages=messages,
                    device=device,
                    native_chat=native_chat,
                )
            )

        except Exception:

            # Do not keep a failed turn in history.
            messages.pop()

            raise

        # ---------------------------------------------------------
        # Empty response
        # ---------------------------------------------------------

        if not response:

            # No assistant response occurred.
            # Remove the user message so the next
            # turn starts from the last valid state.
            messages.pop()

            console.print()

            console.print(
                "[yellow]"
                "Model returned no text."
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
        # NORMAL CHAT OUTPUT
        #
        # No token stats.
        # No panels.
        # No evaluation labels.
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


# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------

def interactive(
    experiment_arg: Path | None,
    model_arg: str | None,
    device_arg: str,
) -> None:

    # -----------------------------------------------------------------
    # Experiment
    # -----------------------------------------------------------------

    if experiment_arg is not None:

        experiment = (
            experiment_arg
            .expanduser()
            .resolve()
        )

    else:

        experiment = (
            choose_experiment()
        )

    (
        manifest,
        original_path,
        transformed_path,
    ) = resolve_models(
        experiment
    )

    # -----------------------------------------------------------------
    # Original / transformed
    # -----------------------------------------------------------------

    selected_model = (
        model_arg
        or choose_model_type()
    )

    if selected_model == "exit":
        return

    if selected_model == "original":

        weights_path = (
            original_path
        )

        label = (
            "Original Model"
        )

    else:

        weights_path = (
            transformed_path
        )

        label = (
            "Transformed Model"
        )

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------

    device = choose_device(
        device_arg
    )

    repo_id = (
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

    console.print()

    console.print(
        "[dim]"
        f"Model: {repo_id}\n"
        f"Experiment: {experiment_name}\n"
        f"Testing: {label}\n"
        f"Device: {device}"
        "[/dim]"
    )

    tokenizer = None
    model = None

    # -----------------------------------------------------------------
    # Load and chat
    # -----------------------------------------------------------------

    try:

        (
            tokenizer,
            model,
            generation_config,
            native_chat,
        ) = load_chat_model(
            weights_path=weights_path,
            reference_path=original_path,
            device=device,
        )

        chat(
            tokenizer=tokenizer,
            model=model,
            generation_config=(
                generation_config
            ),
            native_chat=native_chat,
            device=device,
            label=label,
        )

    finally:

        if model is not None:

            unload_model(
                tokenizer,
                model,
            )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Chat with a WeightLab original "
            "or transformed model."
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
            experiment_arg=(
                args.experiment
            ),
            model_arg=(
                args.model
            ),
            device_arg=(
                args.device
            ),
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