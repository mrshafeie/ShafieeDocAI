"""Classical horizontal-projection text-line segmentation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class TextLine:
    """A detected horizontal text-line region."""

    index: int
    start_y: int
    end_y: int

    @property
    def height(self) -> int:
        return self.end_y - self.start_y

    def to_dict(self) -> dict[str, int]:
        return {
            "index": self.index,
            "start_y": self.start_y,
            "end_y": self.end_y,
            "height": self.height,
        }


@dataclass(frozen=True)
class SegmentationResult:
    """Line segmentation output and diagnostic information."""

    image_width: int
    image_height: int
    threshold: int
    horizontal_projection: tuple[int, ...]
    lines: tuple[TextLine, ...]


def otsu_threshold(image: Image.Image) -> int:
    """Calculate an Otsu threshold from a grayscale image."""

    grayscale = image.convert("L")
    histogram = grayscale.histogram()
    total_pixels = grayscale.width * grayscale.height

    total_intensity = sum(
        value * count
        for value, count in enumerate(histogram)
    )

    background_weight = 0
    background_sum = 0
    best_variance = -1.0
    best_threshold = 127

    for threshold in range(256):
        background_weight += histogram[threshold]
        background_sum += threshold * histogram[threshold]

        foreground_weight = total_pixels - background_weight

        if background_weight == 0:
            continue

        if foreground_weight == 0:
            break

        background_mean = background_sum / background_weight
        foreground_mean = (
            total_intensity - background_sum
        ) / foreground_weight

        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )

        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold

    return best_threshold


def _build_projection(
    image: Image.Image,
    threshold: int,
    row_ink_threshold: int | None,
) -> list[int]:
    """Count dark pixels in every row."""

    grayscale = image.convert("L")
    width, height = grayscale.size
    pixels = grayscale.load()

    minimum_ink = (
        row_ink_threshold
        if row_ink_threshold is not None
        else max(2, round(width * 0.0008))
    )

    projection: list[int] = []

    for y in range(height):
        ink_count = 0

        for x in range(width):
            if pixels[x, y] <= threshold:
                ink_count += 1

        projection.append(
            ink_count if ink_count >= minimum_ink else 0
        )

    return projection


def _merge_small_gaps(
    projection: list[int],
    max_gap: int,
) -> list[int]:
    """Bridge short empty gaps inside a text-line region."""

    if max_gap <= 0:
        return projection.copy()

    merged = projection.copy()
    height = len(merged)
    y = 0

    while y < height:
        if merged[y] > 0:
            y += 1
            continue

        gap_start = y

        while y < height and merged[y] == 0:
            y += 1

        gap_length = y - gap_start

        has_ink_before = gap_start > 0 and merged[gap_start - 1] > 0
        has_ink_after = y < height and merged[y] > 0

        if (
            gap_length <= max_gap
            and has_ink_before
            and has_ink_after
        ):
            for gap_y in range(gap_start, y):
                # Mark the row as connected without inventing
                # a large ink count in the diagnostic profile.
                merged[gap_y] = 1

    return merged


def _extract_ranges(
    projection: list[int],
    min_line_height: int,
    padding: int,
) -> list[tuple[int, int]]:
    """Extract inclusive line boundaries from a row projection."""

    height = len(projection)
    ranges: list[tuple[int, int]] = []
    y = 0

    while y < height:
        if projection[y] == 0:
            y += 1
            continue

        start = y

        while y < height and projection[y] > 0:
            y += 1

        end_exclusive = y
        raw_height = end_exclusive - start

        if raw_height < min_line_height:
            continue

        start_y = max(0, start - padding)
        end_y = min(height, end_exclusive + padding)

        # Merge regions if padding makes them overlap.
        if ranges and start_y <= ranges[-1][1]:
            previous_start, previous_end = ranges[-1]
            ranges[-1] = (
                previous_start,
                max(previous_end, end_y),
            )
        else:
            ranges.append((start_y, end_y))

    return ranges


def segment_lines(
    image: Image.Image,
    *,
    threshold: int | None = None,
    row_ink_threshold: int | None = None,
    max_gap: int = 2,
    padding: int = 2,
    min_line_height: int = 3,
) -> SegmentationResult:
    """
    Detect text-line bands using horizontal projection.

    Pixel values at or below the threshold are treated as ink.
    Coordinates use half-open intervals: [start_y, end_y).
    """

    if image.width < 1 or image.height < 1:
        raise ValueError("Image dimensions must be positive.")

    if threshold is not None and not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255.")

    if row_ink_threshold is not None and row_ink_threshold < 1:
        raise ValueError("row_ink_threshold must be positive.")

    if max_gap < 0:
        raise ValueError("max_gap cannot be negative.")

    if padding < 0:
        raise ValueError("padding cannot be negative.")

    if min_line_height < 1:
        raise ValueError("min_line_height must be positive.")

    grayscale = image.convert("L")
    actual_threshold = (
        otsu_threshold(grayscale)
        if threshold is None
        else threshold
    )

    projection = _build_projection(
        grayscale,
        actual_threshold,
        row_ink_threshold,
    )

    merged_projection = _merge_small_gaps(
        projection,
        max_gap,
    )

    ranges = _extract_ranges(
        merged_projection,
        min_line_height,
        padding,
    )

    lines = tuple(
        TextLine(index=index, start_y=start, end_y=end)
        for index, (start, end) in enumerate(ranges, start=1)
    )

    return SegmentationResult(
        image_width=grayscale.width,
        image_height=grayscale.height,
        threshold=actual_threshold,
        horizontal_projection=tuple(projection),
        lines=lines,
    )


def save_line_crops(
    image: Image.Image,
    result: SegmentationResult,
    output_dir: str | Path,
) -> list[Path]:
    """Save detected line crops as numbered PNG files."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    grayscale = image.convert("L")
    saved_paths: list[Path] = []

    for line in result.lines:
        crop = grayscale.crop(
            (0, line.start_y, result.image_width, line.end_y)
        )

        output_path = destination / f"line_{line.index:03d}.png"
        crop.save(output_path)
        saved_paths.append(output_path)

    return saved_paths


def save_debug_image(
    image: Image.Image,
    result: SegmentationResult,
    output_path: str | Path,
) -> Path:
    """Draw detected line boundaries on a copy of the input image."""

    preview = image.convert("RGB")
    drawing = ImageDraw.Draw(preview)

    for line in result.lines:
        drawing.rectangle(
            (
                0,
                line.start_y,
                result.image_width - 1,
                max(line.start_y, line.end_y - 1),
            ),
            outline=(255, 0, 0),
            width=2,
        )

        drawing.text(
            (5, line.start_y),
            f"Line {line.index}",
            fill=(255, 0, 0),
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    preview.save(destination)

    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Persian document text lines."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--output",
        default="models/line_segmentation",
        help="Directory for extracted line images",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Fixed threshold from 0 to 255; default uses Otsu",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=2,
        help="Maximum empty rows to bridge",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Vertical padding around each line",
    )
    parser.add_argument(
        "--min-line-height",
        type=int,
        default=3,
        help="Ignore regions shorter than this many pixels",
    )
    args = parser.parse_args()

    image_path = Path(args.image)

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    result = segment_lines(
        image,
        threshold=args.threshold,
        max_gap=args.max_gap,
        padding=args.padding,
        min_line_height=args.min_line_height,
    )

    output_dir = Path(args.output)
    saved_paths = save_line_crops(image, result, output_dir)

    debug_path = save_debug_image(
        image,
        result,
        output_dir / "segmentation_debug.png",
    )

    print(f"Image: {image.width}x{image.height}")
    print(f"Threshold: {result.threshold}")
    print(f"Detected lines: {len(result.lines)}")

    for line, path in zip(result.lines, saved_paths):
        print(
            f"Line {line.index}: "
            f"y={line.start_y}..{line.end_y - 1}, "
            f"height={line.height}, "
            f"file={path}"
        )

    print(f"Debug image: {debug_path}")


if __name__ == "__main__":
    main()