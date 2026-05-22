"""Run OCR on each Pascal-VOC object bbox and write XML with <label> per object.

Requires Tesseract on PATH, or set TESSDATA_PREFIX / pass --tesseract-cmd (see
https://github.com/tesseract-ocr/tesseract).

Example:
  python scripts/ocr_voc_annotation.py \\
    --xml Rico/Annotations/8.xml \\
    --image Rico/JPEGImages/8.jpg \\
    --out outputs/8_ocr.xml
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytesseract
from PIL import Image


def _clamp_box(
    xmin: int, ymin: int, xmax: int, ymax: int, w: int, h: int
) -> tuple[int, int, int, int]:
    xmin = max(0, min(w - 1, xmin))
    ymin = max(0, min(h - 1, ymin))
    xmax = max(0, min(w, xmax))
    ymax = max(0, min(h, ymax))
    if xmax <= xmin or ymax <= ymin:
        return xmin, ymin, xmin, ymin
    return xmin, ymin, xmax, ymax


def ocr_crop(image: Image.Image, xmin: int, ymin: int, xmax: int, ymax: int, psm: int) -> str:
    w, h = image.size
    xmin, ymin, xmax, ymax = _clamp_box(xmin, ymin, xmax, ymax, w, h)
    if xmax <= xmin or ymax <= ymin:
        return ""
    crop = image.crop((xmin, ymin, xmax, ymax))
    if crop.width < 2 or crop.height < 2:
        return ""
    config = f"--psm {psm}"
    text = pytesseract.image_to_string(crop, config=config)
    return " ".join(text.split()).strip()


def add_or_replace_label(obj: ET.Element, text: str) -> None:
    for old in list(obj.findall("label")):
        obj.remove(old)
    label_el = ET.Element("label")
    label_el.text = text if text else ""
    children = list(obj)
    bnd_idx = -1
    for i, ch in enumerate(children):
        if ch.tag == "bndbox":
            bnd_idx = i
            break
    if bnd_idx >= 0:
        new_order = children[: bnd_idx + 1] + [label_el] + children[bnd_idx + 1 :]
    else:
        new_order = children + [label_el]
    for ch in list(obj):
        obj.remove(ch)
    for ch in new_order:
        obj.append(ch)


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR each VOC bbox and emit XML with <label>.")
    parser.add_argument("--xml", type=Path, required=True, help="Input Pascal-VOC annotation XML.")
    parser.add_argument("--image", type=Path, required=True, help="Screenshot (same coords as annotation).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output XML path. Default: <xml_stem>_ocr.xml next to input XML.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Path to tesseract executable (default: pytesseract default / PATH).",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Tesseract page segmentation mode (default: 6 — uniform text block).",
    )
    parser.add_argument(
        "--only-name",
        action="append",
        default=None,
        metavar="CLASS",
        help="Only OCR objects whose <name> matches (repeatable). Default: all objects.",
    )
    args = parser.parse_args()

    xml_path = args.xml.resolve()
    img_path = args.image.resolve()
    if not xml_path.is_file():
        raise SystemExit(f"Missing XML: {xml_path}")
    if not img_path.is_file():
        raise SystemExit(f"Missing image: {img_path}")

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    elif shutil.which("tesseract") is None:
        raise SystemExit(
            "tesseract not found on PATH. Install Tesseract OCR or pass --tesseract-cmd "
            "(Windows: often C:\\Program Files\\Tesseract-OCR\\tesseract.exe)."
        )

    out_path = args.out
    if out_path is None:
        out_path = xml_path.parent / f"{xml_path.stem}_ocr.xml"
    else:
        out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    only = None
    if args.only_name:
        only = {x.strip() for x in args.only_name if x.strip()}

    tree = ET.parse(xml_path)
    root = tree.getroot()
    im = Image.open(img_path).convert("RGB")

    count = 0
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        bnd = obj.find("bndbox")
        if bnd is None:
            add_or_replace_label(obj, "")
            continue
        if only is not None and name not in only:
            add_or_replace_label(obj, "")
            continue
        try:
            xmin = int(float(bnd.findtext("xmin", "0")))
            ymin = int(float(bnd.findtext("ymin", "0")))
            xmax = int(float(bnd.findtext("xmax", "0")))
            ymax = int(float(bnd.findtext("ymax", "0")))
        except ValueError:
            add_or_replace_label(obj, "")
            continue
        text = ocr_crop(im, xmin, ymin, xmax, ymax, args.psm)
        add_or_replace_label(obj, text)
        count += 1

    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {out_path} (OCR run on {count} object(s), all have <label>)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
