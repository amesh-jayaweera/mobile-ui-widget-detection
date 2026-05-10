"""Overlay Rico Pascal-VOC boxes + class labels on a screenshot.

Looks up Rico/JPEGImages/{id}.jpg (or .png) and Rico/Annotations/{id}.xml.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _class_color(name: str) -> tuple[int, int, int]:
    h = hashlib.sha256(name.encode()).digest()
    return (h[0], h[1], h[2])


def _find_image(jpeg_dir: Path, stem: str) -> Path:
    for ext in (".jpg", ".jpeg", ".png", ".PNG", ".JPG"):
        p = jpeg_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No image for stem {stem!r} under {jpeg_dir}")


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


def _load_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Draw Rico VOC annotations on an image.")
    parser.add_argument("--rico", type=Path, default=Path("Rico"), help="Rico dataset root.")
    parser.add_argument("--id", default="1070", help="Image id without extension, e.g. 1070.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path. Default: outputs/{id}_rico_overlay.png",
    )
    args = parser.parse_args()

    rico = args.rico.resolve()
    ann = rico / "Annotations" / f"{args.id}.xml"
    if not ann.is_file():
        raise SystemExit(f"Missing annotation: {ann}")

    img_path = _find_image(rico / "JPEGImages", args.id)
    out_path = args.out
    if out_path is None:
        out_path = Path("outputs") / f"{args.id}_rico_overlay.png"
    elif out_path.is_dir():
        out_path = out_path / f"{args.id}_rico_overlay.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    objs = parse_voc_objects(ann)
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _load_font(14)

    for name, xmin, ymin, xmax, ymax in objs:
        color = _class_color(name)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
        label = name
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = xmin, ymin - th - 4
        if ty < 0:
            ty = ymin + 2
        draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill=color)
        draw.text((tx + 2, ty + 2), label, fill=(255, 255, 255), font=font)

    im.save(out_path)
    print(f"Wrote {out_path} ({len(objs)} objects) from {img_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
