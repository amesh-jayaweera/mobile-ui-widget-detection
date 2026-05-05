"""Train YOLO11n on the prepared Rico dataset.

Run `python scripts/prepare_dataset.py` first to populate `data/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Train YOLO11n on Rico UI dataset.")
    parser.add_argument("--data", default="data/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Base weights to fine-tune.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size. Use -1 to let Ultralytics auto-pick based on memory.",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument(
        "--device",
        default="",
        help="Compute device, e.g. 'cpu', '0', or '0,1'. Empty = auto.",
    )
    parser.add_argument("--project", default="runs", help="Output root for runs.")
    parser.add_argument("--name", default="rico_yolo11n", help="Run name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise SystemExit(
            f"data file not found at {data_path}. Run scripts/prepare_dataset.py first."
        )

    from ultralytics import YOLO

    model = YOLO(args.weights)

    train_kwargs = dict(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        cos_lr=True,
        pretrained=True,
        seed=args.seed,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
    if args.device:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    save_dir = getattr(results, "save_dir", None) or Path(args.project) / args.name
    print()
    print(f"Training complete. Best weights: {Path(save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
