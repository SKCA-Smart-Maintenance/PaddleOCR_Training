"""
setup_notebook.py — PaddleOCR environment setup with rich progress output.

Usage in notebook:
    # Download
    !wget -nc https://raw.githubusercontent.com/SKCA-Smart-Maintenance/PaddleOCR_Training/main/utils/setup_notebook.py
    # Import & run
    import setup_notebook
    setup_notebook.setup_paddleocr()
    setup_notebook.check_system()
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List

# ---------------------------------------------------------------------------
# Bootstrap: install 'rich' quietly if missing
# ---------------------------------------------------------------------------
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "rich", "-q"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, label: str = "") -> None:
    """Run a shell command, stream output, and raise on failure."""
    display = label or cmd
    console.print(f"  [dim]$ {cmd}[/dim]")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {display}")


def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=36, style="cyan", complete_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS_BASE: List[dict] = [
    {
        "label": "Clone PaddleOCR repo",
        "fn": lambda: _clone_paddleocr(),
    },
    {
        "label": "Install requirements.txt",
        "fn": lambda: _run(
            "pip install -r PaddleOCR/requirements.txt -q",
            "requirements.txt",
        ),
    },
    {
        "label": "Install paddleocr & paddlex",
        "fn": lambda: _run(
            'pip install -U colored paddleocr "paddleocr[all]" paddlex "paddlex[base]" -q',
            "paddleocr + paddlex",
        ),
    },
    {
        "label": "Install paddle2onnx",
        "fn": lambda: _run("paddlex --install paddle2onnx", "paddle2onnx"),
    },
]

STEPS_GPU: List[dict] = [
    {
        "label": "Install paddlepaddle-gpu (CUDA 12.6)",
        "fn": lambda: _run(
            "pip install paddlepaddle-gpu==3.3.0 "
            "-i https://www.paddlepaddle.org.cn/packages/stable/cu126/ -q",
            "paddlepaddle-gpu",
        ),
    },
    {
        "label": "Install onnxruntime-gpu",
        "fn": lambda: _run("pip install onnxruntime-gpu -q", "onnxruntime-gpu"),
    },
]

STEPS_CPU: List[dict] = [
    {
        "label": "Install paddlepaddle (CPU)",
        "fn": lambda: _run(
            "pip install paddlepaddle==3.3.0 "
            "-i https://www.paddlepaddle.org.cn/packages/stable/cpu/ -q",
            "paddlepaddle-cpu",
        ),
    },
    {
        "label": "Install onnxruntime",
        "fn": lambda: _run("pip install onnxruntime -q", "onnxruntime"),
    },
]


def _clone_paddleocr() -> None:
    if os.path.isdir("PaddleOCR"):
        console.print("  [green]✔[/green] PaddleOCR directory already exists — skipping clone")
        return
    _run("git clone https://github.com/PaddlePaddle/PaddleOCR.git", "git clone")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_paddleocr() -> None:
    """Install PaddleOCR and all its dependencies with a rich progress display."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    backend_label = "GPU [bright_green](CUDA detected)[/]" if cuda_available else "CPU [yellow](no CUDA)[/]"
    steps = STEPS_BASE + (STEPS_GPU if cuda_available else STEPS_CPU)
    total = len(steps)

    console.print(
        Panel.fit(
            f"[bold]PaddleOCR Setup[/bold]\n"
            f"Backend : {backend_label}\n"
            f"Steps   : {total}",
            title="[bold cyan]🚀 Environment Setup[/]",
            border_style="cyan",
        )
    )

    errors: List[str] = []

    with _make_progress() as progress:
        task = progress.add_task("Overall progress", total=total)

        for i, step in enumerate(steps, 1):
            progress.update(task, description=f"[{i}/{total}] {step['label']}")
            console.rule(f"[bold dim]Step {i} — {step['label']}", style="dim")
            try:
                step["fn"]()
                console.print(f"  [bold green]✔ Done[/bold green]: {step['label']}\n")
            except Exception as exc:
                console.print(f"  [bold red]✘ Failed[/bold red]: {step['label']}\n  {exc}\n")
                errors.append(step["label"])
            finally:
                progress.advance(task)

    # ── Summary ────────────────────────────────────────────────────────────
    if errors:
        console.print(
            Panel(
                "\n".join(f"  [red]✘[/red] {e}" for e in errors),
                title="[bold red]⚠ Setup completed with errors[/]",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold green]All steps completed successfully! 🎉[/bold green]",
                border_style="green",
            )
        )


def check_system() -> None:
    """Print a rich summary table of the current environment."""
    console.print(Panel.fit("[bold]System Information[/bold]", border_style="blue"))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold cyan", min_width=20)
    table.add_column("Value", style="white")

    # Python
    table.add_row("Python", sys.version.split()[0])

    # PyTorch / CUDA
    try:
        import torch
        table.add_row("PyTorch", torch.__version__)
        if torch.cuda.is_available():
            table.add_row("CUDA", torch.version.cuda or "unknown")
            table.add_row("GPU", torch.cuda.get_device_name(0))
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            table.add_row("GPU Memory", f"{mem_gb:.1f} GB")
        else:
            table.add_row("GPU", "[yellow]Not available[/yellow]")
    except ImportError:
        table.add_row("PyTorch", "[dim]not installed[/dim]")

    # PaddlePaddle
    try:
        import paddle
        table.add_row("PaddlePaddle", paddle.__version__)
        table.add_row("Paddle device", paddle.device.get_available_device())
    except ImportError:
        table.add_row("PaddlePaddle", "[dim]not installed[/dim]")

    # PaddleOCR
    try:
        import paddleocr
        table.add_row("PaddleOCR", getattr(paddleocr, "__version__", "installed"))
    except ImportError:
        table.add_row("PaddleOCR", "[dim]not installed[/dim]")

    # CPU info
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        table.add_row("CPU", info.get("brand_raw", "unknown"))
        table.add_row("CPU Cores", str(info.get("count", "unknown")))
    except ImportError:
        import platform, multiprocessing
        table.add_row("CPU", platform.processor() or platform.machine())
        table.add_row("CPU Cores", str(multiprocessing.cpu_count()))

    console.print(table)
    console.print()