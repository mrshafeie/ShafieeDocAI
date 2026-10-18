
"""Generate synthetic Persian invoices and their training annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display


GENERATOR_VERSION = "0.1.0"
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1250
BACKGROUND = (255, 255, 255)
FOREGROUND = (30, 30, 30)
MUTED = (100, 100, 100)
BORDER = (190, 190, 190)

SELLERS = [
    "فروشگاه رایانه پارس",
    "تجهیزات اداری آریا",
    "فروشگاه دیجیتال سپهر",
    "بازرگانی نوین شرق",
    "فروشگاه لوازم خانگی امید",
]

BUYERS = [
    "شرکت توسعه نرم افزار",
    "فروشگاه مرکزی تهران",
    "شرکت خدمات رایانه",
    "بازرگانی نمونه",
    "موسسه فناوری ایرانیان",
]

PRODUCTS = [
    ("صفحه کلید", 850_000),
    ("موس رایانه", 420_000),
    ("حافظه جانبی", 1_250_000),
    ("کابل شبکه", 180_000),
    ("مانیتور", 8_500_000),
    ("کاغذ چاپ", 320_000),
    ("فلش مموری", 650_000),
    ("شارژر", 950_000),
    ("هدفون", 1_800_000),
    ("دیسک ذخیره سازی", 2_400_000),
]

PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def persian_number(value: int) -> str:
    """Format an integer using Persian digits and grouping separators."""
    return f"{value:,}".translate(PERSIAN_DIGITS)


def display_text(text: str) -> str:
    """Prepare logical Persian text for rendering with Pillow."""
    return get_display(arabic_reshaper.reshape(text))


def find_font(explicit_path: str | None = None) -> Path:
    """Find a suitable installed font, preferring Tahoma on Windows."""
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    candidates.extend(
        [
            Path("C:/Windows/Fonts/tahoma.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeui.ttf"),
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "No suitable font was found. Pass the font path with --font."
    )


class InvoiceRenderer:
    """Render an invoice and collect the bounding box of every text line."""

    def __init__(self, font_path: Path) -> None:
        self.image = Image.new(
            "RGB",
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            BACKGROUND,
        )
        self.draw = ImageDraw.Draw(self.image)
        self.font_path = font_path
        self.annotations: list[dict[str, Any]] = []
        self.next_annotation_id = 1
        self.y = 45

        self.title_font = ImageFont.truetype(str(font_path), 34)
        self.heading_font = ImageFont.truetype(str(font_path), 25)
        self.body_font = ImageFont.truetype(str(font_path), 21)
        self.small_font = ImageFont.truetype(str(font_path), 18)

    def add_text(
        self,
        text: str,
        *,
        font: ImageFont.FreeTypeFont | None = None,
        color: tuple[int, int, int] = FOREGROUND,
        right: int = 930,
        gap: int = 16,
    ) -> str:
        """Draw one RTL text line and return its annotation ID."""
        font = font or self.body_font
        rendered = display_text(text)

        # Right-top anchoring makes the text position predictable.
        bbox = self.draw.textbbox(
            (right, self.y),
            rendered,
            font=font,
            anchor="rt",
        )

        x1, y1, x2, y2 = bbox

        if x1 < 20 or x2 > IMAGE_WIDTH - 20:
            raise ValueError(
                f"Text does not fit on the invoice: {text!r}. "
                "Reduce the font size or shorten the text."
            )

        self.draw.text(
            (right, self.y),
            rendered,
            font=font,
            fill=color,
            anchor="rt",
        )

        annotation_id = f"text-{self.next_annotation_id:04d}"
        self.next_annotation_id += 1

        self.annotations.append(
            {
                "annotation_id": annotation_id,
                "text": text,
                "bbox": [x1, y1, x2, y2],
                "unit": "line",
                "reading_order": len(self.annotations),
                "text_type": "printed",
                "transcription_status": "verified",
                "normalized_text": None,
            }
        )

        self.y = y2 + gap
        return annotation_id

    def add_separator(self, gap: int = 15) -> None:
        """Draw a separator that is not a text annotation."""
        self.draw.line(
            (60, self.y, 940, self.y),
            fill=BORDER,
            width=2,
        )
        self.y += gap

    def finish(self, output_path: Path) -> None:
        """Save the generated invoice image."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(output_path, format="PNG", optimize=True)


def make_invoice(
    rng: random.Random,
    index: int,
    font_path: Path,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    """Generate one invoice image and its JSONL annotation record."""
    renderer = InvoiceRenderer(font_path)

    invoice_number = f"INV-{index:06d}"
    invoice_date = date(
        1405,
        rng.randint(1, 12),
        rng.randint(1, 28),
    ).strftime("%Y/%m/%d").translate(PERSIAN_DIGITS)

    seller = rng.choice(SELLERS)
    buyer = rng.choice(BUYERS)

    selected_products = rng.sample(PRODUCTS, k=rng.randint(2, 5))
    items: list[dict[str, Any]] = []
    subtotal = 0

    renderer.add_text(
        "فاکتور فروش",
        font=renderer.title_font,
        gap=22,
    )
    renderer.add_separator()

    invoice_number_annotation = renderer.add_text(
        f"شماره فاکتور: {invoice_number}",
        font=renderer.heading_font,
    )
    date_annotation = renderer.add_text(
        f"تاریخ فاکتور: {invoice_date}",
    )
    seller_annotation = renderer.add_text(f"فروشنده: {seller}")
    buyer_annotation = renderer.add_text(f"خریدار: {buyer}")

    renderer.add_separator()
    renderer.add_text("شرح کالا | تعداد | قیمت واحد | مبلغ", font=renderer.heading_font)
    renderer.add_separator(gap=12)

    for product_name, unit_price in selected_products:
        quantity = rng.randint(1, 5)
        line_total = quantity * unit_price
        subtotal += line_total

        line_text = (
            f"{product_name} | "
            f"{persian_number(quantity)} | "
            f"{persian_number(unit_price)} | "
            f"{persian_number(line_total)}"
        )

        annotation_id = renderer.add_text(
            line_text,
            font=renderer.small_font,
            gap=17,
        )

        items.append(
            {
                "description": product_name,
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": line_total,
                "source_annotation_ids": [annotation_id],
            }
        )

    tax = subtotal * 10 // 100
    total = subtotal + tax

    renderer.add_separator()
    subtotal_annotation = renderer.add_text(
        f"جمع کالاها: {persian_number(subtotal)} تومان",
        font=renderer.heading_font,
    )
    tax_annotation = renderer.add_text(
        f"مالیات: {persian_number(tax)} تومان",
    )
    total_annotation = renderer.add_text(
        f"مبلغ نهایی: {persian_number(total)} تومان",
        font=renderer.heading_font,
        gap=25,
    )

    renderer.add_text(
        "این فاکتور به صورت مصنوعی تولید شده است.",
        font=renderer.small_font,
        color=MUTED,
    )

    image_relative_path = Path("images") / f"{invoice_number}.png"
    image_path = output_dir / image_relative_path

    renderer.finish(image_path)

    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

    def field(value: Any, annotation_id: str) -> dict[str, Any]:
        return {
            "value": value,
            "status": "verified",
            "source_annotation_ids": [annotation_id],
        }

    return {
        "schema_version": "1.0",
        "document_id": invoice_number,
        "document_type": "invoice",
        "language": "fa",
        "image": {
            "path": image_relative_path.as_posix(),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "sha256": image_hash,
        },
        "split": split,
        "text_annotations": renderer.annotations,
        "fields": {
            "invoice_number": field(invoice_number, invoice_number_annotation),
            "invoice_date": field(invoice_date, date_annotation),
            "seller_name": field(seller, seller_annotation),
            "buyer_name": field(buyer, buyer_annotation),
            "subtotal": field(subtotal, subtotal_annotation),
            "tax": field(tax, tax_annotation),
            "total": field(total, total_annotation),
        },
        "items": items,
        "metadata": {
            "source_type": "synthetic",
            "annotation_status": "verified",
            "generator_version": GENERATOR_VERSION,
        },
    }


def assign_split(index: int, count: int) -> str:
    """Assign records deterministically to train, validation, and test."""
    train_end = int(count * 0.8)
    validation_end = int(count * 0.9)

    if index < train_end:
        return "train"

    if index < validation_end:
        return "validation"

    return "test"


def generate_dataset(
    count: int,
    output_dir: str | Path,
    seed: int = 42,
    font: str | None = None,
) -> Path:
    """Generate invoices and write a JSONL dataset."""
    if count < 1:
        raise ValueError("count must be at least 1.")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    font_path = find_font(font)
    rng = random.Random(seed)

    dataset_path = destination / "dataset.jsonl"

    with dataset_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index in range(count):
            split = assign_split(index, count)

            record = make_invoice(
                rng=rng,
                index=index + 1,
                font_path=font_path,
                output_dir=destination,
                split=split,
            )

            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    return dataset_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Persian invoice images and annotations."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of invoices to generate (default: 100).",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic",
        help="Output directory (default: data/synthetic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Optional path to a TrueType font.",
    )

    args = parser.parse_args()

    try:
        dataset_path = generate_dataset(
            count=args.count,
            output_dir=args.output,
            seed=args.seed,
            font=args.font,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Generated {args.count} invoices.")
    print(f"Dataset: {dataset_path}")
    print(f"Images: {dataset_path.parent / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
