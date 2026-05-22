"""Overlay Pascal-VOC boxes with numeric IDs 1..n and three selectable colors.

Takes explicit --image and --xml paths. Default color is red; set GREEN_IDS and
BLUE_IDS below (1-based indices matching object order in the XML) to recolor.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Color assignment: 1-based object indices as they appear in the XML.
# IDs listed here use that color; all others use RED.
# ---------------------------------------------------------------------------
GREEN_IDS: list[int] = []
BLUE_IDS: list[int] = []

_COLORS = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (40, 100, 220),
}


def parse_voc_objects(xml_path: Path) -> list[tuple[str, int, int, int, int]]:
    root = ET.parse(xml_path).getroot()
    out: list[tuple[str, int, int, int, int]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if not name or bnd is None:
            continue
        xmin = int(float(bnd.findtext("xmin", "0")))
        ymin = int(float(bnd.findtext("ymin", "0")))
        xmax = int(float(bnd.findtext("xmax", "0")))
        ymax = int(float(bnd.findtext("ymax", "0")))
        out.append((name, xmin, ymin, xmax, ymax))
    return out


def _load_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _parse_id_list(s: str | None) -> list[int]:
    if not s or not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw VOC boxes with numeric labels 1..n (red/green/blue)."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--xml", type=Path, required=True, help="Pascal VOC XML path.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image. Default: outputs/{xml_stem}_numbered.png",
    )
    parser.add_argument(
        "--green-ids",
        default="",
        help='Optional comma-separated 1-based IDs for green (overrides GREEN_IDS from file).',
    )
    parser.add_argument(
        "--blue-ids",
        default="",
        help='Optional comma-separated 1-based IDs for blue (overrides BLUE_IDS from file).',
    )
    args = parser.parse_args()

    img_path = args.image.resolve()
    xml_path = args.xml.resolve()
    if not img_path.is_file():
        raise SystemExit(f"Missing image: {img_path}")
    if not xml_path.is_file():
        raise SystemExit(f"Missing XML: {xml_path}")

    green_ids = _parse_id_list(args.green_ids) if args.green_ids else list(GREEN_IDS)
    blue_ids = _parse_id_list(args.blue_ids) if args.blue_ids else list(BLUE_IDS)

    def color_for_index(i: int) -> tuple[int, int, int]:
        if i in green_ids:
            return _COLORS["green"]
        if i in blue_ids:
            return _COLORS["blue"]
        return _COLORS["red"]

    out_path = args.out
    if out_path is None:
        out_path = Path("outputs") / f"{xml_path.stem}_numbered.png"
    else:
        out_path = out_path.resolve()
    if out_path.is_dir():
        out_path = out_path / f"{xml_path.stem}_numbered.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    objs = parse_voc_objects(xml_path)
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _load_font(22)

    for idx, (_name, xmin, ymin, xmax, ymax) in enumerate(objs, start=1):
        color = color_for_index(idx)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        label = str(idx)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = xmin, ymin - th - 6
        if ty < 0:
            ty = ymin + 4
        draw.rectangle([tx, ty, tx + tw + 8, ty + th + 6], fill=color)
        draw.text((tx + 4, ty + 3), label, fill=(255, 255, 255), font=font)

    im.save(out_path)
    print(f"Wrote {out_path} ({len(objs)} objects) from {img_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
