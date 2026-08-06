"""Build a compact contact sheet from per-sample GLI QA PNG files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--cell-width", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no QA PNG files found in {args.input_dir}")
    if args.columns <= 0 or args.cell_width <= 0:
        raise ValueError("columns and cell-width must be positive")
    images = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        target_height = round(image.height * args.cell_width / image.width)
        images.append(image.resize((args.cell_width, target_height), Image.Resampling.LANCZOS))
    cell_height = max(image.height for image in images)
    title_height = 72
    rows = math.ceil(len(images) / args.columns)
    canvas = Image.new(
        "RGB",
        (args.columns * args.cell_width, title_height + rows * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), f"{args.title} - {len(images)} frozen validation QA samples", fill="black")
    for index, image in enumerate(images):
        x = (index % args.columns) * args.cell_width
        y = title_height + (index // args.columns) * cell_height
        canvas.paste(image, (x, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
