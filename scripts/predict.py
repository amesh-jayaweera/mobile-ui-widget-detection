"""Run a trained YOLO11 model on screenshots and emit Rico-schema VOC XML.

Usage:
    python scripts/predict.py \
        --weights runs/rico_yolo11n/weights/best.pt \
        --source path/to/image_or_directory \
        --out predictions/

Each output XML mirrors the schema used in `Rico/Annotations/*.xml`:
    <annotation>
      <folder>JPEGImages</folder>
      <filename>...</filename>
      <size><width/><height/><depth>3</depth></size>
      <object>
        <name/><difficult>0</difficult>
        <bndbox><xmin/><ymin/><xmax/><ymax/></bndbox>
      </object>
      ...
    </annotation>
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _clip(value: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(value))))


def build_voc_xml(
    filename: str,
    width: int,
    height: int,
    detections: list[tuple[str, float, float, float, float, float]],
) -> ET.ElementTree:
    """Construct a Pascal VOC XML tree matching the Rico annotation schema."""
    root = ET.Element("annotation")

    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = filename

    size_el = ET.SubElement(root, "size")
    ET.SubElement(size_el, "width").text = str(width)
    ET.SubElement(size_el, "height").text = str(height)
    ET.SubElement(size_el, "depth").text = "3"

    for name, xmin, ymin, xmax, ymax, _conf in detections:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(_clip(xmin, 0, max(0, width - 1)))
        ET.SubElement(bnd, "ymin").text = str(_clip(ymin, 0, max(0, height - 1)))
        ET.SubElement(bnd, "xmax").text = str(_clip(xmax, 0, max(0, width - 1)))
        ET.SubElement(bnd, "ymax").text = str(_clip(ymax, 0, max(0, height - 1)))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    return tree


def collect_sources(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(
            p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise SystemExit(f"--source not found: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict UI elements and write VOC XML.")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights (.pt).")
    parser.add_argument("--source", required=True, help="Image file or directory of images.")
    parser.add_argument("--out", required=True, help="Directory to write XML files into.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="", help="Compute device, e.g. 'cpu' or '0'.")
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Also save annotated preview images alongside the XMLs.",
    )
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = collect_sources(source_path)
    if not sources:
        print(f"warn: no images found under {source_path}")
        return 0

    from ultralytics import YOLO

    model = YOLO(args.weights)
    class_names: dict[int, str] = dict(model.names) if hasattr(model, "names") else {}

    predict_kwargs = dict(
        source=[str(p) for p in sources],
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
        save=args.save_vis,
        project=str(out_dir),
        name="vis" if args.save_vis else "_tmp",
        exist_ok=True,
    )
    if args.device:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)

    written = 0
    for r in results:
        img_path = Path(r.path)
        h, w = r.orig_shape

        detections: list[tuple[str, float, float, float, float, float]] = []
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            for (xmin, ymin, xmax, ymax), cid, c in zip(xyxy, cls, confs):
                name = class_names.get(int(cid), str(int(cid)))
                detections.append((name, float(xmin), float(ymin), float(xmax), float(ymax), float(c)))

        tree = build_voc_xml(img_path.name, int(w), int(h), detections)
        xml_path = out_dir / f"{img_path.stem}.xml"
        tree.write(xml_path, encoding="utf-8", xml_declaration=False)
        written += 1
        print(f"  {img_path.name} -> {xml_path.name}  ({len(detections)} objects)")

    print()
    print(f"Wrote {written} XML file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
