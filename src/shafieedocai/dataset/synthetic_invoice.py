"""Generate synthetic Persian invoices with template-level dataset splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

GENERATOR_VERSION = "0.2.0"
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1250
BACKGROUND = (255, 255, 255)
FOREGROUND = (30, 30, 30)
MUTED = (100, 100, 100)
BORDER = (190, 190, 190)
SPLIT_NAMES = ("train", "validation", "test")
TEMPLATE_COUNT = 10

SELLERS = [
    "فروشگاه رایانه پارس", "تجهیزات اداری آریا", "فروشگاه دیجیتال سپهر",
    "بازرگانی نوین شرق", "فروشگاه لوازم خانگی امید",
]
BUYERS = [
    "شرکت توسعه نرم افزار", "فروشگاه مرکزی تهران", "شرکت خدمات رایانه",
    "بازرگانی نمونه", "موسسه فناوری ایرانیان",
]
PRODUCTS = [
    ("صفحه کلید", 850_000), ("موس رایانه", 420_000),
    ("حافظه جانبی", 1_250_000), ("کابل شبکه", 180_000),
    ("مانیتور", 8_500_000), ("کاغذ چاپ", 320_000),
    ("فلش مموری", 650_000), ("شارژر", 950_000),
    ("هدفون", 1_800_000), ("دیسک ذخیره سازی", 2_400_000),
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
    candidates.extend([
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("No suitable font was found. Pass the font path with --font.")


def template_split_map(seed: int = 42, template_count: int = TEMPLATE_COUNT) -> dict[str, str]:
    """Assign each template ID to exactly one split, deterministically.

    Ten templates yield an 8/1/1 template allocation. This is a practical
    initial split, not a substitute for evaluating more independent templates.
    """
    if template_count < 3:
        raise ValueError("template_count must be at least 3 to populate all splits.")
    ids = list(range(template_count))
    random.Random(seed).shuffle(ids)
    train_count = max(1, int(template_count * 0.8))
    validation_count = max(1, int(template_count * 0.1))
    if train_count + validation_count >= template_count:
        train_count = template_count - 2
        validation_count = 1
    result: dict[str, str] = {}
    for position, template_id in enumerate(ids):
        if position < train_count:
            split = "train"
        elif position < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        result[f"invoice-template-{template_id:02d}"] = split
    return result


class InvoiceRenderer:
    """Render one RTL invoice and capture line-level text annotations."""

    def __init__(self, font_path: Path, template_id: int) -> None:
        self.image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BACKGROUND)
        self.draw = ImageDraw.Draw(self.image)
        self.font_path = font_path
        self.template_id = template_id
        self.annotations: list[dict[str, Any]] = []
        self.next_annotation_id = 1
        # Real structural differences, not just different invoice values.
        self.margin = 45 + (template_id % 3) * 12
        self.right = 930 - (template_id % 2) * 25
        self.gap_scale = (0, 3, 6, 9, 12)[template_id % 5]
        self.title_font = ImageFont.truetype(str(font_path), 34 + (template_id % 3) * 2)
        self.heading_font = ImageFont.truetype(str(font_path), 24 + (template_id % 2) * 2)
        self.body_font = ImageFont.truetype(str(font_path), 20 + (template_id % 3))
        self.small_font = ImageFont.truetype(str(font_path), 17 + (template_id % 2))
        self.y = self.margin
        self.accent = [
            (70, 90, 125), (40, 100, 90), (125, 75, 70), (90, 80, 125),
            (100, 100, 100),
        ][template_id % 5]

    def add_text(
        self,
        text: str,
        *,
        font: ImageFont.FreeTypeFont | None = None,
        color: tuple[int, int, int] = FOREGROUND,
        right: int | None = None,
        gap: int = 16,
    ) -> str:
        """Draw one RTL line and return its annotation ID."""
        font = font or self.body_font
        right = self.right if right is None else right
        rendered = display_text(text)
        bbox = self.draw.textbbox((right, self.y), rendered, font=font, anchor="rt")
        x1, y1, x2, y2 = bbox
        if x1 < 20 or x2 > IMAGE_WIDTH - 20 or y1 < 0 or y2 > IMAGE_HEIGHT - 10:
            raise ValueError(f"Text does not fit on the invoice: {text!r}.")
        self.draw.text((right, self.y), rendered, font=font, fill=color, anchor="rt")
        annotation_id = f"text-{self.next_annotation_id:04d}"
        self.next_annotation_id += 1
        self.annotations.append({
            "annotation_id": annotation_id,
            "text": text,
            "bbox": [x1, y1, x2, y2],
            "unit": "line",
            "reading_order": len(self.annotations),
            "text_type": "printed",
            "transcription_status": "verified",
            "normalized_text": None,
        })
        self.y = y2 + gap + self.gap_scale
        return annotation_id

    def add_separator(self, gap: int = 15) -> None:
        """Draw a template-specific separator."""
        if self.template_id % 3 == 0:
            self.draw.rectangle((self.margin, self.y, self.right, self.y + 2), fill=self.accent)
            self.y += 2
        elif self.template_id % 3 == 1:
            self.draw.line((self.margin, self.y, self.right, self.y), fill=self.accent, width=3)
        else:
            self.draw.line((self.margin, self.y, self.right, self.y), fill=BORDER, width=1)
        self.y += gap + self.gap_scale

    def finish(self, output_path: Path) -> None:
        """Save the generated invoice image."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(output_path, format="PNG", optimize=True)


def make_invoice(
    rng: random.Random,
    index: int,
    font_path: Path,
    output_dir: Path,
    template_id: int,
    split: str,
) -> dict[str, Any]:
    """Generate one invoice and its JSONL annotation record."""
    renderer = InvoiceRenderer(font_path, template_id)
    invoice_number = f"INV-{index:06d}"
    invoice_date = date(1405, rng.randint(1, 12), rng.randint(1, 28)).strftime("%Y/%m/%d").translate(PERSIAN_DIGITS)
    seller = rng.choice(SELLERS)
    buyer = rng.choice(BUYERS)
    selected_products = rng.sample(PRODUCTS, k=rng.randint(2, 5))
    items: list[dict[str, Any]] = []
    subtotal = 0

    # Template variation includes title placement and accent/line style.
    title_right = renderer.right if template_id % 2 == 0 else IMAGE_WIDTH // 2 + 120
    renderer.add_text("فاکتور فروش", font=renderer.title_font, color=renderer.accent, right=title_right, gap=22)
    renderer.add_separator()
    invoice_number_annotation = renderer.add_text(f"شماره فاکتور: {invoice_number}", font=renderer.heading_font)
    date_annotation = renderer.add_text(f"تاریخ فاکتور: {invoice_date}")
    seller_annotation = renderer.add_text(f"فروشنده: {seller}")
    buyer_annotation = renderer.add_text(f"خریدار: {buyer}")
    renderer.add_separator()
    renderer.add_text("شرح کالا | تعداد | قیمت واحد | مبلغ", font=renderer.heading_font, color=renderer.accent)
    renderer.add_separator(gap=12)

    for product_name, unit_price in selected_products:
        quantity = rng.randint(1, 5)
        line_total = quantity * unit_price
        subtotal += line_total
        line_text = f"{product_name} | {persian_number(quantity)} | {persian_number(unit_price)} | {persian_number(line_total)}"
        annotation_id = renderer.add_text(line_text, font=renderer.small_font, gap=17)
        items.append({
            "description": product_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "line_total": line_total,
            "source_annotation_ids": [annotation_id],
        })

    tax = subtotal * 10 // 100
    total = subtotal + tax
    renderer.add_separator()
    subtotal_annotation = renderer.add_text(f"جمع کالاها: {persian_number(subtotal)} تومان", font=renderer.heading_font)
    tax_annotation = renderer.add_text(f"مالیات: {persian_number(tax)} تومان")
    total_annotation = renderer.add_text(f"مبلغ نهایی: {persian_number(total)} تومان", font=renderer.heading_font, gap=25)
    renderer.add_text("این فاکتور به صورت مصنوعی تولید شده است.", font=renderer.small_font, color=MUTED)

    image_relative_path = Path("images") / f"{invoice_number}.png"
    image_path = output_dir / image_relative_path
    renderer.finish(image_path)
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

    def field(value: Any, annotation_id: str) -> dict[str, Any]:
        return {"value": value, "status": "verified", "source_annotation_ids": [annotation_id]}

    template_name = f"invoice-template-{template_id:02d}"
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
            "template_id": template_name,
        },
    }


def assign_split(index: int, count: int) -> str:
    """Legacy compatibility helper; new generation splits by template."""
    if count < 1:
        raise ValueError("count must be at least 1.")
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
    """Generate invoices with deterministic template-level split assignment."""
    if count < 1:
        raise ValueError("count must be at least 1.")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    font_path = find_font(font)
    rng = random.Random(seed)
    split_map = template_split_map(seed=seed)
    dataset_path = destination / "dataset.jsonl"

    with dataset_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index in range(count):
            # Cycle through templates so small datasets can exercise multiple splits.
            template_id = index % TEMPLATE_COUNT
            template_name = f"invoice-template-{template_id:02d}"
            split = split_map[template_name]
            record = make_invoice(
                rng=rng,
                index=index + 1,
                font_path=font_path,
                output_dir=destination,
                template_id=template_id,
                split=split,
            )
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return dataset_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic Persian invoice images and annotations.")
    parser.add_argument("--count", type=int, default=100, help="Number of invoices to generate (default: 100).")
    parser.add_argument("--output", default="data/synthetic", help="Output directory (default: data/synthetic).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--font", default=None, help="Optional path to a TrueType font.")
    args = parser.parse_args()
    try:
        dataset_path = generate_dataset(args.count, args.output, seed=args.seed, font=args.font)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {args.count} invoices.")
    print(f"Dataset: {dataset_path}")
    print(f"Images: {dataset_path.parent / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
