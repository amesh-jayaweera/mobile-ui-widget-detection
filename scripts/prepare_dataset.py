"""Convert the Rico Pascal-VOC dataset into a YOLO-format dataset.

- Reads XML annotations from `Rico/Annotations` and JPEGs from `Rico/JPEGImages`.
- Drops rare classes (less than 50 instances in the corpus): Multi_Tab,
  Bottom_Navigation, Spinner, Map.
- Writes YOLO txt labels and a deterministic 80/20 train/val split into
  `data/images/{train,val}` and `data/labels/{train,val}`, plus `data/data.yaml`.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

CLASSES: list[str] = [
    "Text",
    "Icon",
    "Image",
    "TextButton",
    "UpperTaskBar",
    "PageIndicator",
    "CheckedTextView",
    "EditText",
    "BackgroundImage",
    "Modal",
    "Toolbar",
    "Drawer",
    "Switch",
    "Card",
]

DROPPED: set[str] = {"Multi_Tab", "Bottom_Navigation", "Spinner", "Map"}

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Return (width, height, [(name, xmin, ymin, xmax, ymax), ...])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path} has no <size>")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"{xml_path} has invalid size {width}x{height}")

    objs: list[tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            continue
        try:
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))
        except ValueError:
            continue
        objs.append((name, xmin, ymin, xmax, ymax))

    return width, height, objs


def to_yolo_line(name: str, xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int) -> str | None:
    if name not in CLASS_TO_ID:
        return None

    xmin = max(0.0, min(float(w), xmin))
    ymin = max(0.0, min(float(h), ymin))
    xmax = max(0.0, min(float(w), xmax))
    ymax = max(0.0, min(float(h), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None

    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.0, min(1.0, bw))
    bh = max(0.0, min(1.0, bh))

    cls_id = CLASS_TO_ID[name]
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from Rico VOC.")
    parser.add_argument("--rico", default="Rico", help="Path to the Rico folder.")
    parser.add_argument("--out", default="data", help="Output dataset folder.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--link",
        choices=["copy", "symlink"],
        default="copy",
        help="How to place images into the dataset folder. 'copy' is safest on Windows.",
    )
    args = parser.parse_args()

    rico_root = Path(args.rico).resolve()
    ann_dir = rico_root / "Annotations"
    img_dir = rico_root / "JPEGImages"
    out_root = Path(args.out).resolve()

    if not ann_dir.is_dir() or not img_dir.is_dir():
        print(f"ERROR: expected {ann_dir} and {img_dir} to exist.", file=sys.stderr)
        return 1

    xml_files = sorted(p for p in ann_dir.glob("*.xml"))
    if not xml_files:
        print(f"ERROR: no XML files under {ann_dir}", file=sys.stderr)
        return 1

    pairs: list[tuple[Path, Path]] = []
    skipped_missing_image = 0
    for xml_path in xml_files:
        stem = xml_path.stem
        jpg_path = img_dir / f"{stem}.jpg"
        if not jpg_path.exists():
            skipped_missing_image += 1
            continue
        pairs.append((xml_path, jpg_path))

    if skipped_missing_image:
        print(f"warn: {skipped_missing_image} XML files had no matching JPEG and were skipped")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_count = int(round(len(pairs) * args.val_frac))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    splits = {"train": train_pairs, "val": val_pairs}

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = out_root / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    per_split_counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    empty_label_files = 0
    bad_xml_files = 0

    for split_name, split_pairs in splits.items():
        img_out_dir = out_root / "images" / split_name
        lbl_out_dir = out_root / "labels" / split_name

        for xml_path, jpg_path in split_pairs:
            try:
                w, h, objs = parse_voc_xml(xml_path)
            except (ET.ParseError, ValueError) as exc:
                bad_xml_files += 1
                print(f"warn: skipping {xml_path.name}: {exc}")
                continue

            lines: list[str] = []
            for name, xmin, ymin, xmax, ymax in objs:
                line = to_yolo_line(name, xmin, ymin, xmax, ymax, w, h)
                if line is None:
                    continue
                lines.append(line)
                per_split_counts[split_name][name] += 1

            if not lines:
                empty_label_files += 1

            label_path = lbl_out_dir / f"{xml_path.stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            dst_img = img_out_dir / jpg_path.name
            if args.link == "symlink":
                try:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    dst_img.symlink_to(jpg_path)
                except OSError:
                    shutil.copy2(jpg_path, dst_img)
            else:
                shutil.copy2(jpg_path, dst_img)

    yaml_path = out_root / "data.yaml"
    yaml_lines = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, name in enumerate(CLASSES):
        yaml_lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print()
    print(f"Wrote dataset to {out_root}")
    print(f"  train images: {len(train_pairs)}")
    print(f"  val   images: {len(val_pairs)}")
    print(f"  empty-label images (kept as background): {empty_label_files}")
    if bad_xml_files:
        print(f"  malformed XML files skipped: {bad_xml_files}")
    print()
    print("Per-class instance counts:")
    header = f"{'class':<18}{'train':>8}{'val':>8}{'total':>8}"
    print(header)
    print("-" * len(header))
    for name in CLASSES:
        t = per_split_counts["train"][name]
        v = per_split_counts["val"][name]
        print(f"{name:<18}{t:>8}{v:>8}{t + v:>8}")
    print()
    print(f"Dropped classes (filtered out): {sorted(DROPPED)}")
    print(f"Wrote {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
