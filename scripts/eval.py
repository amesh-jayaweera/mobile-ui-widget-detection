"""Evaluate a trained YOLO11 model on the Rico val set.

Runs `model.val(...)` with `plots=True`, which causes Ultralytics to write
the standard evaluation visualizations as PNGs into the run directory:

    P_curve.png                      precision vs confidence
    R_curve.png                      recall vs confidence
    F1_curve.png                     F1 vs confidence
    PR_curve.png                     per-class precision-recall
    confusion_matrix.png             raw counts
    confusion_matrix_normalized.png  row-normalized

Usage:
    python scripts/eval.py --weights runs/rico_yolo11n/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PLOT_FILES = (
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png",
    "PR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 on Rico val split.")
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights.")
    parser.add_argument("--data", default="data/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--split", default="val", choices=("val", "test", "train"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001, help="Conf threshold for metrics.")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold.")
    parser.add_argument("--device", default="", help="Compute device, e.g. 'cpu' or '0'.")
    parser.add_argument("--project", default="runs", help="Output root.")
    parser.add_argument("--name", default="rico_yolo11n_eval", help="Run name.")
    parser.add_argument(
        "--copy-to",
        default="",
        help="Optional directory to also copy the generated PNGs into.",
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    weights_path = Path(args.weights).resolve()
    if not data_path.exists():
        raise SystemExit(f"data file not found: {data_path}")
    if not weights_path.exists():
        raise SystemExit(f"weights not found: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    val_kwargs = dict(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=False,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    if args.device:
        val_kwargs["device"] = args.device

    metrics = model.val(**val_kwargs)

    save_dir = Path(getattr(metrics, "save_dir", Path(args.project) / args.name))

    print()
    print(f"Evaluation complete. Plots written to: {save_dir}")
    print()
    print("Saved PNGs:")
    for name in PLOT_FILES:
        p = save_dir / name
        marker = "  ok " if p.exists() else "  -- "
        print(f"{marker}{p}")

    if hasattr(metrics, "box"):
        b = metrics.box
        print()
        print("Aggregate metrics (val):")
        print(f"  precision (mean): {float(b.mp):.4f}")
        print(f"  recall    (mean): {float(b.mr):.4f}")
        print(f"  mAP@0.5         : {float(b.map50):.4f}")
        print(f"  mAP@0.5:0.95    : {float(b.map):.4f}")

    if args.copy_to:
        dst_dir = Path(args.copy_to).resolve()
        dst_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for name in PLOT_FILES:
            src = save_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)
                copied += 1
        print()
        print(f"Copied {copied}/{len(PLOT_FILES)} PNG(s) to {dst_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
