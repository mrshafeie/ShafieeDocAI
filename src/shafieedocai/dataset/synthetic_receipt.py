"""Generate synthetic Persian warehouse receipts using a fixed layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont


GENERATOR_VERSION = "0.1.0"
TEMPLATE_ID = "warehouse-receipt-template-01"

IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 1100

WHITE = (255, 255, 255)
BLACK = (35, 35, 35)
GRAY = (110, 110, 110)
BORDER = (115, 115, 115)

PERSIAN_DIGITS = str.maketrans(
    "0123456789",
    "۰۱۲۳۴۵۶۷۸۹",
)

SELLERS = [
    "شرکت تجهیزات صنعتی نمونه",
    "فروشگاه ابزار و ماشین آریا",
    "بازرگانی قطعات صنعتی پارس",
    "تجهیزات فنی سپهر",
    "فروشگاه لوازم صنعتی نوین",
]

BUYERS = [
    "شرکت فنی مهندسی نمونه",
    "کارگاه صنعتی آریا",
    "مجموعه خدمات فنی پارس",
    "شرکت تولیدی سپهر",
    "انبار مرکزی تجهیزات نوین",
]

PRODUCTS = [
    ("پیچ و مهره صنعتی", "۴۱۰", "عدد"),
    ("واشر فلزی", "۴۲۰", "بسته"),
    ("بلبرینگ صنعتی", "۴۳۰", "عدد"),
    ("کابل برق", "۴۴۰", "متر"),
    ("بست فلزی", "۴۵۰", "عدد"),
    ("لوله صنعتی", "۴۶۰", "شاخه"),
    ("اتصال فلزی", "۴۷۰", "عدد"),
    ("ابزار تعمیراتی", "۴۸۰", "عدد"),
    ("قطعه یدکی صنعتی", "۴۹۰", "عدد"),
    ("رول نوار صنعتی", "۵۰۰", "رول"),
]

SELLER_ADDRESSES = [
    "تبریز، شهرک صنعتی، خیابان اصلی",
    "تهران، خیابان صنعت، پلاک ۱۲",
    "اصفهان، منطقه صنعتی، خیابان سوم",
    "کرج، شهرک صنعتی، بلوار کارگاه",
]

BUYER_ADDRESSES = [
    "تبریز، خیابان کارگاه، پلاک ۲۱",
    "تهران، منطقه صنعتی، خیابان دوم",
    "اصفهان، خیابان انبار، پلاک ۸",
    "کرج، بلوار تولید، خیابان چهارم",
]


def fa_number(value: int) -> str:
    """Convert an integer to Persian digits."""
    return str(value).translate(PERSIAN_DIGITS)


def display_text(text: str) -> str:
    """Shape Persian text and apply bidirectional display ordering."""
    return get_display(arabic_reshaper.reshape(str(text)))


def find_font(explicit_path: str | None = None) -> Path:
    """Find a font that supports Persian characters."""
    candidates: list[Path] = []

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
        "Persian font not found. Specify a font using --font."
    )


class ReceiptRenderer:
    """Render a fixed-layout receipt and record text annotations."""

    def __init__(self, font_path: Path) -> None:
        self.image = Image.new(
            "RGB",
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            WHITE,
        )
        self.draw = ImageDraw.Draw(self.image)

        self.title_font = ImageFont.truetype(str(font_path), 30)
        self.heading_font = ImageFont.truetype(str(font_path), 22)
        self.body_font = ImageFont.truetype(str(font_path), 18)
        self.small_font = ImageFont.truetype(str(font_path), 15)

        self.annotations: list[dict[str, Any]] = []
        self.next_id = 1

    def text(
        self,
        value: str,
        right: int,
        top: int,
        font: ImageFont.FreeTypeFont | None = None,
        color: tuple[int, int, int] = BLACK,
    ) -> str:
        """Draw text and save its bounding box and transcription."""
        font = font or self.body_font
        rendered = display_text(value)

        bbox = self.draw.textbbox(
            (right, top),
            rendered,
            font=font,
            anchor="rt",
        )

        x1, y1, x2, y2 = bbox

        if x1 < 10 or x2 >= IMAGE_WIDTH - 5:
            raise ValueError(
                f"Text does not fit in the receipt: {value!r}"
            )

        self.draw.text(
            (right, top),
            rendered,
            font=font,
            fill=color,
            anchor="rt",
        )

        annotation_id = f"text-{self.next_id:05d}"
        self.next_id += 1

        self.annotations.append(
            {
                "annotation_id": annotation_id,
                "text": value,
                "bbox": [x1, y1, x2, y2],
                "unit": "line",
                "reading_order": len(self.annotations),
                "text_type": "printed",
                "transcription_status": "verified",
                "normalized_text": None,
            }
        )

        return annotation_id

    def rectangle(
        self,
        box: tuple[int, int, int, int],
        width: int = 2,
    ) -> None:
        self.draw.rectangle(
            box,
            outline=BORDER,
            width=width,
        )

    def line(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int = 1,
    ) -> None:
        self.draw.line(
            (x1, y1, x2, y2),
            fill=BORDER,
            width=width,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path, format="PNG", optimize=True)


def make_receipt(
    rng: random.Random,
    index: int,
    font_path: Path,
    output_dir: Path,
    split: str,
) -> dict[str, Any]:
    """Generate one synthetic receipt and its structured annotations."""
    renderer = ReceiptRenderer(font_path)

    document_number = f"WH-{index:06d}"
    document_date = (
        f"۱۴۰۵/{fa_number(rng.randint(1, 12)).zfill(2)}/"
        f"{fa_number(rng.randint(1, 28)).zfill(2)}"
    )

    seller = rng.choice(SELLERS)
    buyer = rng.choice(BUYERS)
    seller_address = rng.choice(SELLER_ADDRESSES)
    buyer_address = rng.choice(BUYER_ADDRESSES)

    # Synthetic contact numbers; never copied from the real sample.
    seller_phone = f"۰۹۱۲{rng.randint(1000000, 9999999)}"
    buyer_phone = f"۰۹۳۵{rng.randint(1000000, 9999999)}"

    field_annotations: dict[str, str] = {}

    # Title and document metadata.
    renderer.rectangle((345, 35, 655, 95), width=3)
    renderer.text(
        "رسید انبار",
        right=610,
        top=48,
        font=renderer.title_font,
    )

    field_annotations["document_number"] = renderer.text(
        f"شماره رسید: {document_number}",
        right=930,
        top=112,
        font=renderer.small_font,
    )
    field_annotations["document_date"] = renderer.text(
        f"تاریخ رسید: {document_date}",
        right=930,
        top=142,
        font=renderer.small_font,
    )

    # Seller information panel.
    renderer.rectangle((55, 180, 945, 290))
    renderer.rectangle((830, 180, 945, 290))

    renderer.text(
        "فروشنده",
        right=930,
        top=190,
        font=renderer.heading_font,
    )

    field_annotations["seller_name"] = renderer.text(
        f"نام و نام خانوادگی: {seller}",
        right=805,
        top=195,
        font=renderer.small_font,
    )
    field_annotations["seller_phone"] = renderer.text(
        f"تلفن: {seller_phone}",
        right=805,
        top=228,
        font=renderer.small_font,
    )
    field_annotations["seller_address"] = renderer.text(
        f"آدرس: {seller_address}",
        right=805,
        top=258,
        font=renderer.small_font,
    )

    # Buyer information panel.
    renderer.rectangle((55, 305, 945, 415))
    renderer.rectangle((830, 305, 945, 415))

    renderer.text(
        "خریدار",
        right=930,
        top=315,
        font=renderer.heading_font,
    )

    field_annotations["buyer_name"] = renderer.text(
        f"نام و نام خانوادگی: {buyer}",
        right=805,
        top=320,
        font=renderer.small_font,
    )
    field_annotations["buyer_phone"] = renderer.text(
        f"تلفن: {buyer_phone}",
        right=805,
        top=353,
        font=renderer.small_font,
    )
    field_annotations["buyer_address"] = renderer.text(
        f"آدرس: {buyer_address}",
        right=805,
        top=383,
        font=renderer.small_font,
    )

    # Goods table. Coordinates are arranged from right to left.
    table_left = 55
    table_right = 945
    table_top = 445
    header_bottom = 490
    row_height = 48

    columns = [
        (875, 945),  # Row number
        (760, 875),  # Code
        (465, 760),  # Description
        (340, 465),  # Quantity
        (250, 340),  # Unit
    ]

    renderer.rectangle(
        (table_left, table_top, table_right, header_bottom)
    )

    headers = [
        ("ردیف", 930),
        ("کد", 860),
        ("شرح", 735),
        ("تعداد", 450),
        ("واحد", 325),
    ]

    for label, right in headers:
        renderer.text(
            label,
            right=right,
            top=456,
            font=renderer.small_font,
        )

    for x, _ in columns[1:]:
        renderer.line(x, table_top, x, header_bottom)

    selected_products = rng.sample(
        PRODUCTS,
        k=rng.randint(2, 4),
    )

    items: list[dict[str, Any]] = []

    for row_index, (description, code, unit) in enumerate(
        selected_products,
        start=1,
    ):
        y1 = header_bottom + (row_index - 1) * row_height
        y2 = y1 + row_height

        renderer.rectangle(
            (table_left, y1, table_right, y2)
        )

        for x, _ in columns[1:]:
            renderer.line(x, y1, x, y2)

        quantity = rng.randint(1, 20)

        row_annotations = [
            renderer.text(
                fa_number(row_index),
                right=930,
                top=y1 + 10,
                font=renderer.small_font,
            ),
            renderer.text(
                code,
                right=860,
                top=y1 + 10,
                font=renderer.small_font,
            ),
            renderer.text(
                description,
                right=735,
                top=y1 + 10,
                font=renderer.small_font,
            ),
            renderer.text(
                fa_number(quantity),
                right=450,
                top=y1 + 10,
                font=renderer.small_font,
            ),
            renderer.text(
                unit,
                right=325,
                top=y1 + 10,
                font=renderer.small_font,
            ),
        ]

        items.append(
            {
                "description": description,
                "code": code,
                "quantity": quantity,
                "unit": unit,
                "source_annotation_ids": row_annotations,
            }
        )

    table_bottom = header_bottom + len(items) * row_height

    # Additional notes and signatures, as in the reference layout.
    notes_top = table_bottom + 30

    field_annotations["description"] = renderer.text(
        "توضیحات: تحویل اقلام مطابق جدول فوق انجام شد.",
        right=930,
        top=notes_top,
        font=renderer.small_font,
    )

    renderer.text(
        "از خرید شما متشکریم",
        right=600,
        top=notes_top + 65,
        font=renderer.body_font,
        color=GRAY,
    )

    renderer.text(
        "مهر و امضای فروشنده:",
        right=900,
        top=notes_top + 125,
        font=renderer.small_font,
    )
    renderer.text(
        "مهر و امضای خریدار:",
        right=400,
        top=notes_top + 125,
        font=renderer.small_font,
    )

    image_relative_path = Path("images") / f"{document_number}.png"
    image_path = output_dir / image_relative_path
    renderer.save(image_path)

    image_hash = hashlib.sha256(
        image_path.read_bytes()
    ).hexdigest()

    def field(value: Any, annotation_id: str) -> dict[str, Any]:
        return {
            "value": value,
            "status": "verified",
            "source_annotation_ids": [annotation_id],
        }

    return {
        "schema_version": "1.0",
        "document_id": document_number,
        "document_type": "warehouse_receipt",
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
            "document_number": field(
                document_number,
                field_annotations["document_number"],
            ),
            "document_date": field(
                document_date,
                field_annotations["document_date"],
            ),
            "seller_name": field(
                seller,
                field_annotations["seller_name"],
            ),
            "seller_phone": field(
                seller_phone,
                field_annotations["seller_phone"],
            ),
            "seller_address": field(
                seller_address,
                field_annotations["seller_address"],
            ),
            "buyer_name": field(
                buyer,
                field_annotations["buyer_name"],
            ),
            "buyer_phone": field(
                buyer_phone,
                field_annotations["buyer_phone"],
            ),
            "buyer_address": field(
                buyer_address,
                field_annotations["buyer_address"],
            ),
            "description": field(
                "تحویل اقلام مطابق جدول فوق انجام شد.",
                field_annotations["description"],
            ),
        },
        "items": items,
        "metadata": {
            "source_type": "synthetic",
            "annotation_status": "verified",
            "generator_version": GENERATOR_VERSION,
            "template_id": TEMPLATE_ID,
        },
    }


def assign_split(index: int, count: int) -> str:
    """Split this single-template dataset by document."""
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
    """Generate receipt images and write their annotations to JSONL."""
    if count < 1:
        raise ValueError("count must be at least 1.")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    font_path = find_font(font)
    rng = random.Random(seed)

    dataset_path = destination / "dataset.jsonl"

    with dataset_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for index in range(count):
            record = make_receipt(
                rng=rng,
                index=index + 1,
                font_path=font_path,
                output_dir=destination,
                split=assign_split(index, count),
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
        description=(
            "Generate synthetic Persian warehouse receipts "
            "using a fixed document layout."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of receipts to generate.",
    )
    parser.add_argument(
        "--output",
        default="data/synthetic_receipts",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
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

    print(f"Generated {args.count} synthetic receipts.")
    print(f"Dataset: {dataset_path}")
    print(f"Images: {dataset_path.parent / 'images'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())