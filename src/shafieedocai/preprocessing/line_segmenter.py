
"""Classical Persian document line segmentation using OpenCV."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class TextLine:
    index: int
    start_y: int
    end_y: int

    @property
    def height(self) -> int:
        return self.end_y - self.start_y


@dataclass(frozen=True)
class SegmentationResult:
    image_width: int
    image_height: int
    lines: tuple[TextLine, ...]
    horizontal_projection: tuple[int, ...]


def binarize(gray: np.ndarray) -> np.ndarray:
    """Create a binary foreground mask using adaptive thresholding."""
    if gray.ndim != 2:
        raise ValueError("Expected a grayscale image.")

    height, width = gray.shape
    if height < 3 or width < 3:
        raise ValueError("Image is too small.")

    # Keep the tested parameters for the 960x1280 photograph.
    block_size = min(51, min(height, width))
    if block_size % 2 == 0:
        block_size -= 1
    block_size = max(3, block_size)

    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        9,
    )


def remove_horizontal_rules(binary: np.ndarray) -> np.ndarray:
    """Remove long horizontal rules on a copy of the binary image."""
    height, width = binary.shape

    # Detect only relatively long horizontal structures.
    kernel_width = max(40, width // 5)
    kernel_width = min(kernel_width, width)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_width, 1),
    )

    horizontal_rules = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    # Extend detected rules slightly to cover broken segments.
    horizontal_rules = cv2.dilate(
        horizontal_rules,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
        iterations=1,
    )

    return cv2.subtract(binary, horizontal_rules)


def horizontal_projection(binary: np.ndarray) -> np.ndarray:
    """Count foreground pixels in each image row."""
    return np.count_nonzero(binary, axis=1).astype(np.int32)


def smooth_projection(
    projection: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Smooth small fluctuations without filling large line gaps."""
    if window_size <= 1:
        return projection.astype(np.float64)

    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(
        projection.astype(np.float64),
        kernel,
        mode="same",
    )


def find_line_ranges(
    projection: np.ndarray,
    image_width: int,
    *,
    min_line_height: int = 5,
    max_gap: int = 3,
    row_threshold: int | None = None,
) -> list[tuple[int, int]]:
    """Find line bands using an explicit row-ink threshold."""
    if image_width <= 0:
        raise ValueError("image_width must be positive.")
    if min_line_height < 1:
        raise ValueError("min_line_height must be positive.")
    if max_gap < 0:
        raise ValueError("max_gap cannot be negative.")

    if projection.size == 0 or int(projection.max()) == 0:
        return []

    if row_threshold is None:
        # For a 960-pixel-wide image this starts at 4 pixels.
        # Unlike a fraction of the peak, it does not force weak noise
        # rows to become active just because one row is very dark.
        row_threshold = max(4, round(image_width * 0.004))

    active = projection >= row_threshold
    height = len(active)

    # Bridge only very short interruptions inside a text band.
    y = 0
    while y < height:
        if active[y]:
            y += 1
            continue

        gap_start = y
        while y < height and not active[y]:
            y += 1
        gap_end = y

        gap_length = gap_end - gap_start
        if (
            0 < gap_length <= max_gap
            and gap_start > 0
            and gap_end < height
            and active[gap_start - 1]
            and active[gap_end]
        ):
            active[gap_start:gap_end] = True

    ranges: list[tuple[int, int]] = []
    y = 0

    while y < height:
        if not active[y]:
            y += 1
            continue

        start = y
        while y < height and active[y]:
            y += 1
        end = y

        if end - start >= min_line_height:
            ranges.append((start, end))

    return ranges


def segment_lines(
    image: np.ndarray,
    *,
    min_line_height: int = 5,
    max_gap: int = 3,
    padding: int = 3,
    row_threshold: int | None = None,
    remove_rules: bool = True,
) -> tuple[SegmentationResult, np.ndarray, np.ndarray]:
    """Return line ranges, original binary mask and cleaned mask."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if padding < 0:
        raise ValueError("padding cannot be negative.")

    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Expected grayscale, BGR, or BGRA image.")

    height, width = gray.shape

    # Step 1: Use the standard adaptive threshold that was tested.
    binary = binarize(gray)

    # Step 2: Remove long horizontal table rules from a copy only.
    cleaned = (
        remove_horizontal_rules(binary)
        if remove_rules
        else binary.copy()
    )

    # Step 3: Count foreground pixels row by row.
    raw_projection = horizontal_projection(cleaned)
    projection = smooth_projection(raw_projection, window_size=3)

    # Step 4: Find text bands using a pixel-count threshold.
    ranges = find_line_ranges(
        projection,
        width,
        min_line_height=min_line_height,
        max_gap=max_gap,
        row_threshold=row_threshold,
    )

    # Step 5: Add padding without letting neighboring crops overlap.
    padded_ranges: list[tuple[int, int]] = []

    for start, end in ranges:
        padded_start = max(0, start - padding)
        padded_end = min(height, end + padding)

        if padded_ranges and padded_start < padded_ranges[-1][1]:
            previous_start, previous_end = padded_ranges[-1]
            midpoint = (previous_end + padded_start) // 2
            padded_ranges[-1] = (previous_start, midpoint)
            padded_start = midpoint

        padded_ranges.append((padded_start, padded_end))

    lines = tuple(
        TextLine(index, start, end)
        for index, (start, end) in enumerate(padded_ranges, start=1)
    )

    result = SegmentationResult(
        image_width=width,
        image_height=height,
        lines=lines,
        horizontal_projection=tuple(int(value) for value in raw_projection),
    )

    return result, binary, cleaned


def save_outputs(
    image: np.ndarray,
    result: SegmentationResult,
    binary: np.ndarray,
    cleaned: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Save line crops and diagnostic images."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]
    debug = image.copy()

    for line in result.lines:
        y1 = max(0, min(height - 1, line.start_y))
        y2 = max(y1, min(height - 1, line.end_y - 1))

        cv2.rectangle(debug, (0, y1), (width - 1, y2), (0, 0, 255), 2)
        cv2.putText(
            debug,
            str(line.index),
            (8, min(height - 1, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        crop = image[line.start_y:line.end_y, :]
        cv2.imwrite(str(output / f"line_{line.index:03d}.png"), crop)

    cv2.imwrite(str(output / "binary.png"), binary)
    cv2.imwrite(str(output / "cleaned_binary.png"), cleaned)
    cv2.imwrite(str(output / "segmentation_debug.png"), debug)

    report = {
        "image_width": result.image_width,
        "image_height": result.image_height,
        "detected_lines": len(result.lines),
        "lines": [
            {**asdict(line), "height": line.height}
            for line in result.lines
        ],
    }

    (output / "segmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classical Persian document line segmentation."
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument(
        "--output",
        default="models/line_segmentation/classical",
        help="Output directory",
    )
    parser.add_argument("--min-line-height", type=int, default=5)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument(
        "--row-threshold",
        type=int,
        default=None,
        help="Minimum foreground pixels required to activate a row",
    )
    parser.add_argument(
        "--keep-rules",
        action="store_true",
        help="Keep horizontal table rules",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Could not read image: {args.image}")

    result, binary, cleaned = segment_lines(
        image,
        min_line_height=args.min_line_height,
        max_gap=args.max_gap,
        padding=args.padding,
        row_threshold=args.row_threshold,
        remove_rules=not args.keep_rules,
    )

    save_outputs(image, result, binary, cleaned, args.output)

    print(f"Image: {result.image_width}x{result.image_height}")
    print("Threshold mode: adaptive (block size 51, C=9 for this image)")
    print(f"Detected lines: {len(result.lines)}")

    for line in result.lines:
        print(
            f"Line {line.index}: "
            f"y={line.start_y}..{line.end_y - 1}, "
            f"height={line.height}"
        )

    print(f"Output directory: {args.output}")
    print(f"Debug image: {Path(args.output) / 'segmentation_debug.png'}")


if __name__ == "__main__":
    main()
