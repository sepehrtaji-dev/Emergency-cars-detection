import argparse
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text
from rich import box
from ultralytics import YOLO

console = Console()

def print_config(cfg: dict):
    table = Table(box=box.ROUNDED, show_header=False, border_style="cyan", padding=(0, 2))
    table.add_column("Key",   style="bold cyan",  no_wrap=True)
    table.add_column("Value", style="white")
    for k, v in cfg.items():
        table.add_row(k, str(v))
    console.print(Panel(table, title="[bold cyan]Training Config[/bold cyan]", border_style="cyan"))

def print_metrics(metrics):
    table = Table(box=box.SIMPLE, border_style="green", padding=(0, 2))
    table.add_column("Metric",   style="bold green")
    table.add_column("Value",    style="bright_white")

    map50    = metrics.box.map50
    map5095  = metrics.box.map
    prec     = metrics.box.mp
    rec      = metrics.box.mr

    def bar(val, width=20):
        filled = int(val * width)
        return "[green]" + "█" * filled + "[/green][dim]" + "░" * (width - filled) + "[/dim]"

    table.add_row("mAP@50",    f"{map50:.4f}   {bar(map50)}")
    table.add_row("mAP@50-95", f"{map5095:.4f}   {bar(map5095)}")
    table.add_row("Precision", f"{prec:.4f}   {bar(prec)}")
    table.add_row("Recall",    f"{rec:.4f}   {bar(rec)}")

    console.print(Panel(table, title="[bold green]Validation Results[/bold green]", border_style="green"))

def train(
    data="dataset/dataset.yaml",
    model="yolov8n.pt",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
    patience=10,
    device="",
    project="runs/detect",
    name="emergency_car",
    resume=False,
):
    cfg = {
        "Model":    model,
        "Data":     data,
        "Epochs":   epochs,
        "Img Size": imgsz,
        "Batch":    batch,
        "LR":       lr0,
        "Patience": patience,
        "Device":   device or "auto",
        "Project":  project,
        "Run Name": name,
        "Resume":   resume,
    }

    console.print()
    console.print(Panel(
        Text("Emergency Car Detector", style="bold white", justify="center"),
        subtitle="[dim]YOLOv8 Training[/dim]",
        border_style="cyan",
        padding=(1, 4),
    ))

    print_config(cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        yolo = YOLO(model)
        progress.update(task, description="Model loaded.")

    console.print("[bold cyan]Starting training...[/bold cyan]\n")

    start = time.time()

    results = yolo.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        patience=patience,
        device=device if device else None,
        project=project,
        name=name,
        resume=resume,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.1,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        plots=True,
        save=True,
        verbose=True,
    )

    elapsed = time.time() - start
    console.print(f"\n[dim]Training finished in {elapsed/60:.1f} minutes[/dim]\n")

    console.print("[bold green]Validating...[/bold green]\n")
    metrics = yolo.val(data=data)
    print_metrics(metrics)

    best = Path(project) / name / "weights" / "best.pt"
    console.print(Panel(
        f"[bold white]Best weights:[/bold white] [cyan]{best.resolve()}[/cyan]\n"
        f"[bold white]Run:[/bold white]          [dim]python predict.py --source your_image.jpg[/dim]",
        title="[bold green]Done![/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default="dataset/dataset.yaml", help="Path to dataset config file")
    parser.add_argument("--model",    default="yolov8n.pt",        help="Path to model file")
    parser.add_argument("--epochs",   default=50,                  type=int, help="Number of training epochs")
    parser.add_argument("--imgsz",    default=640,                 type=int, help="Image size")
    parser.add_argument("--batch",    default=16,                  type=int, help="Batch size")
    parser.add_argument("--lr0",      default=0.01,                type=float, help="Initial learning rate")
    parser.add_argument("--patience", default=10,                  type=int, help="Patience for early stopping")
    parser.add_argument("--device",   default="",                   help="Device to use (CPU or GPU)")
    parser.add_argument("--project",  default="runs/detect",      help="Project name")
    parser.add_argument("--name",     default="emergency_car",    help="Name of the run")
    parser.add_argument("--resume",   action="store_true", help="Resume training from the last checkpoint")
    args = parser.parse_args()

    train(**vars(args))