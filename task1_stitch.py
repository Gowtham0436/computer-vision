"""
Task 1 – Panorama stitching utility.

This script implements the high-level OpenCV Stitcher API (see the bundled
"OpenCV_ High level stitching API (Stitcher class).pdf") to stitch a series of
overlapping photos into a panorama.  It also provides an optional comparison
against a reference panorama (for example, the panorama produced on a mobile
device) by generating a side-by-side contact sheet.

Typical usage (from the repository root):

    python task1_stitch.py \\
        --images ./images \\
        --output ./output/task1_panorama1.jpg \\
        --reference ./comparisons/mobi_panaroma.JPG \\
        --comparison-output ./output/task1_vs_mobi.jpg

Requirements:
    - Python 3.9+
    - OpenCV with the contrib stitching module
      (pip install opencv-python OR opencv-contrib-python)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch overlapping images using OpenCV's Stitcher API."
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("images"),
        help="Directory that contains the ordered input photos (default: ./images).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.JPG",
        help="Glob pattern (relative to --images) used to collect photos (default: *.JPG).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/task1_panorama1.jpg"),
        help="Destination file for the stitched panorama.",
    )
    parser.add_argument(
        "--resize-max-width",
        type=int,
        default=1800,
        help=(
            "Optional down-sampling width applied to each frame before stitching. "
            "Set to 0 to keep the original resolution."
        ),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional reference panorama (e.g., from a mobile phone) used for comparison.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=Path("output/task1_vs_mobi.jpg"),
        help="Where to write the side-by-side comparison (only if --reference is provided).",
    )
    parser.add_argument(
        "--exposure-compensation",
        type=str,
        default="gain_blocks",
        choices=["none", "gain", "gain_blocks", "channel", "channel_blocks"],
        help="Exposure compensation strategy passed to the Stitcher.",
    )
    return parser.parse_args(argv)


def collect_image_paths(folder: Path, pattern: str) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    paths = sorted(folder.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No input files matched {pattern} inside {folder}")
    return paths


def load_images(
    paths: Iterable[Path], max_width: int = 0, color_flag: int = cv2.IMREAD_COLOR
) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for path in paths:
        img = cv2.imread(str(path), color_flag)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        if max_width and img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_size = (max_width, int(img.shape[0] * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        images.append(img)
    return images


def create_stitcher() -> cv2.Stitcher:
    mode = cv2.Stitcher_PANORAMA
    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(mode)
    elif hasattr(cv2, "createStitcher"):
        stitcher = cv2.createStitcher(mode)
    else:
        raise RuntimeError("This version of OpenCV does not expose the Stitcher API.")
    return stitcher


def set_exposure_compensation(stitcher: cv2.Stitcher, strategy: str) -> None:
    """
    Attempt to set exposure compensation on the stitcher.
    
    Note: The high-level cv2.Stitcher API doesn't expose setExposureCompensator.
    This function gracefully skips exposure compensation if the method isn't available.
    The stitcher will use its default exposure compensation settings.
    """
    # Check if the method exists (it won't on the high-level Stitcher API)
    if not hasattr(stitcher, "setExposureCompensator"):
        # High-level API doesn't support this - use default compensation
        return
    if not hasattr(cv2, "detail") or not hasattr(cv2.detail, "ExposureCompensator"):
        return
    compensator_map = {
        "none": cv2.detail.ExposureCompensator_NO,
        "gain": cv2.detail.ExposureCompensator_GAIN,
        "gain_blocks": cv2.detail.ExposureCompensator_GAIN_BLOCKS,
        "channel": cv2.detail.ExposureCompensator_CHANNELS,
        "channel_blocks": cv2.detail.ExposureCompensator_CHANNELS_BLOCKS,
    }
    stitcher.setExposureCompensator(cv2.detail.ExposureCompensator_createDefault(
        compensator_map[strategy]
    ))


def stitch_images(
    images: Sequence[np.ndarray],
    exposure_strategy: str = "gain_blocks",
) -> np.ndarray:
    if len(images) < 2:
        raise ValueError("Need at least two images to perform stitching.")
    stitcher = create_stitcher()
    set_exposure_compensation(stitcher, exposure_strategy)
    status, panorama = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"Stitching failed with status code {status}")
    return panorama


def make_side_by_side(
    stitched: np.ndarray,
    reference_path: Path,
    output_path: Path,
) -> Path:
    ref = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if ref is None:
        raise ValueError(f"Could not read reference panorama: {reference_path}")

    # Normalize heights for a fair comparison.
    target_height = min(stitched.shape[0], ref.shape[0])

    def resize_to_height(image: np.ndarray) -> np.ndarray:
        scale = target_height / image.shape[0]
        if scale == 1.0:
            return image
        new_size = (int(image.shape[1] * scale), target_height)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    stitched_resized = resize_to_height(stitched)
    ref_resized = resize_to_height(ref)

    padding = 20
    pad = np.full((target_height, padding, 3), 255, dtype=np.uint8)
    comparison = np.hstack([ref_resized, pad, stitched_resized])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), comparison)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    image_paths = collect_image_paths(args.images, args.pattern)
    images = load_images(image_paths, max_width=max(0, args.resize_max_width))

    print(f"Loaded {len(images)} frames for stitching.")
    panorama = stitch_images(images, args.exposure_compensation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), panorama)
    print(f"Panorama saved to {args.output.resolve()}")

    if args.reference:
        comparison_path = make_side_by_side(panorama, args.reference, args.comparison_output)
        print(f"Comparison saved to {comparison_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

