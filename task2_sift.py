"""
Assignment 4 – Task 2
======================

Implement SIFT feature extraction from scratch (simplified) and compare the
results with OpenCV's reference implementation.  The script exposes a CLI that
takes two overlapping photos, computes keypoints/descriptors using both the
custom pipeline and OpenCV's SIFT, performs descriptor matching followed by
RANSAC-based homography estimation, and finally writes visualisations of the
matches.

This implementation purposefully keeps the code self-contained: NumPy is used
for linear algebra, and OpenCV is used only for image I/O, resizing, and a few
primitive operations (Gaussian blur, colour conversion, drawing).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures


@dataclasses.dataclass
class Keypoint:
    """Minimal representation of a SIFT keypoint used in this assignment."""

    x: float
    y: float
    octave: int
    layer: int
    sigma: float
    orientation: float


@dataclasses.dataclass
class Match:
    idx_a: int
    idx_b: int
    distance: float


# ---------------------------------------------------------------------------
# Utility helpers


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task 2 – SIFT from scratch with RANSAC comparison"
    )
    parser.add_argument("--image-a", type=Path, required=True, help="First image")
    parser.add_argument("--image-b", type=Path, required=True, help="Second image")
    parser.add_argument(
        "--resize-width",
        type=int,
        default=960,
        help="Optional width to resize both images to (keeps aspect ratio)",
    )
    parser.add_argument(
        "--octaves", type=int, default=4, help="Number of octaves in the pyramid"
    )
    parser.add_argument(
        "--scales",
        type=int,
        default=3,
        help="Number of scales (per octave) for SIFT Gaussian pyramid",
    )
    parser.add_argument(
        "--contrast-threshold",
        type=float,
        default=0.04,
        help="Contrast threshold used to discard weak extrema",
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=10.0,
        help="R parameter used to suppress edge responses",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.6,
        help="Base blur applied to the images before building the pyramid",
    )
    parser.add_argument(
        "--ratio-test",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold for descriptor matching",
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=2000,
        help="RANSAC iterations used when estimating homography",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=3.0,
        help="Inlier threshold (pixels) used during RANSAC",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/task2"),
        help="Directory that will store diagnostic artefacts",
    )
    return parser.parse_args(argv)


def load_image(path: Path, resize_width: int | None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    if resize_width and image.shape[1] > resize_width:
        scale = resize_width / image.shape[1]
        new_size = (resize_width, int(round(image.shape[0] * scale)))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def to_grayscale_float(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Custom SIFT implementation (simplified)


class SIFTFromScratch:
    def __init__(
        self,
        num_octaves: int = 4,
        num_scales: int = 3,
        sigma: float = 1.6,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
    ) -> None:
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold

    # ------------------------ Public API ---------------------------------

    def detect_and_compute(
        self, image_gray: np.ndarray
    ) -> Tuple[List[Keypoint], np.ndarray]:
        base = cv2.GaussianBlur(image_gray, (0, 0), self.sigma, borderType=cv2.BORDER_REPLICATE)
        gaussian_pyramid = self._build_gaussian_pyramid(base)
        dog_pyramid = self._build_dog_pyramid(gaussian_pyramid)
        keypoints = self._find_scale_space_extrema(gaussian_pyramid, dog_pyramid)
        oriented_keypoints = self._assign_orientations(keypoints, gaussian_pyramid)
        descriptors = self._compute_descriptors(oriented_keypoints, gaussian_pyramid)
        return oriented_keypoints, descriptors

    # ----------------------- Pyramid construction ------------------------

    def _build_gaussian_pyramid(self, base: np.ndarray) -> List[List[np.ndarray]]:
        pyramid: List[List[np.ndarray]] = []
        k = 2 ** (1 / self.num_scales)
        sigma0 = self.sigma

        for octave_idx in range(self.num_octaves):
            octave_images: List[np.ndarray] = []
            sigma_prev = sigma0
            octave_images.append(base)
            for scale_idx in range(1, self.num_scales + 3):
                sigma_total = sigma0 * (k ** scale_idx)
                sigma_diff = math.sqrt(max(sigma_total**2 - sigma_prev**2, 1e-6))
                blurred = cv2.GaussianBlur(
                    octave_images[-1],
                    (0, 0),
                    sigma_diff,
                    borderType=cv2.BORDER_REPLICATE,
                )
                octave_images.append(blurred)
                sigma_prev = sigma_total
            pyramid.append(octave_images)

            # Prepare base for next octave (downsample by factor of 2)
            next_base = octave_images[-3]
            height, width = next_base.shape
            if height <= 16 or width <= 16:
                break
            base = cv2.resize(
                next_base,
                (width // 2, height // 2),
                interpolation=cv2.INTER_NEAREST,
            )

        return pyramid

    def _build_dog_pyramid(
        self, gaussian_pyramid: List[List[np.ndarray]]
    ) -> List[List[np.ndarray]]:
        dog_pyramid: List[List[np.ndarray]] = []
        for octave in gaussian_pyramid:
            dog_octave = []
            for i in range(1, len(octave)):
                dog_octave.append(octave[i] - octave[i - 1])
            dog_pyramid.append(dog_octave)
        return dog_pyramid

    # ----------------------- Keypoint detection --------------------------

    def _find_scale_space_extrema(
        self,
        gaussian_pyramid: List[List[np.ndarray]],
        dog_pyramid: List[List[np.ndarray]],
    ) -> List[Keypoint]:
        keypoints: List[Keypoint] = []
        threshold = self.contrast_threshold / self.num_scales

        for octave_idx, dog_octave in enumerate(dog_pyramid):
            for layer_idx in range(1, len(dog_octave) - 1):
                prev_img = dog_octave[layer_idx - 1]
                curr_img = dog_octave[layer_idx]
                next_img = dog_octave[layer_idx + 1]
                rows, cols = curr_img.shape
                for y in range(1, rows - 1):
                    for x in range(1, cols - 1):
                        value = curr_img[y, x]
                        if abs(value) < threshold:
                            continue
                        patch = np.concatenate(
                            [
                                prev_img[y - 1 : y + 2, x - 1 : x + 2].ravel(),
                                curr_img[y - 1 : y + 2, x - 1 : x + 2].ravel(),
                                next_img[y - 1 : y + 2, x - 1 : x + 2,].ravel(),
                            ]
                        )
                        if value > 0 and value != patch.max():
                            continue
                        if value < 0 and value != patch.min():
                            continue
                        if self._is_edge_response(curr_img, x, y):
                            continue
                        sigma = self.sigma * (2 ** octave_idx) * (2 ** (layer_idx / self.num_scales))
                        kp = Keypoint(
                            x=x * (2**octave_idx),
                            y=y * (2**octave_idx),
                            octave=octave_idx,
                            layer=layer_idx,
                            sigma=sigma,
                            orientation=0.0,
                        )
                        keypoints.append(kp)

        return keypoints

    def _is_edge_response(self, image: np.ndarray, x: int, y: int) -> bool:
        dxx = image[y, x + 1] + image[y, x - 1] - 2 * image[y, x]
        dyy = image[y + 1, x] + image[y - 1, x] - 2 * image[y, x]
        dxy = (
            image[y + 1, x + 1]
            + image[y - 1, x - 1]
            - image[y + 1, x - 1]
            - image[y - 1, x + 1]
        )
        tr = dxx + dyy
        det = dxx * dyy - dxy**2
        if det <= 0:
            return True
        r = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        return (tr * tr) * r >= det

    # ----------------------- Orientation assignment ----------------------

    def _assign_orientations(
        self, keypoints: List[Keypoint], gaussian_pyramid: List[List[np.ndarray]]
    ) -> List[Keypoint]:
        oriented: List[Keypoint] = []
        for kp in keypoints:
            octave_images = gaussian_pyramid[kp.octave]
            gaussian_img = octave_images[kp.layer]
            scale = kp.sigma
            radius = int(round(3 * scale))
            weight_factor = -0.5 / (scale**2)
            hist = np.zeros(36, dtype=np.float32)

            x = int(round(kp.x / (2**kp.octave)))
            y = int(round(kp.y / (2**kp.octave)))

            rows, cols = gaussian_img.shape
            for dy in range(-radius, radius + 1):
                yy = y + dy
                if yy <= 0 or yy >= rows - 1:
                    continue
                for dx in range(-radius, radius + 1):
                    xx = x + dx
                    if xx <= 0 or xx >= cols - 1:
                        continue
                    gx = gaussian_img[yy, xx + 1] - gaussian_img[yy, xx - 1]
                    gy = gaussian_img[yy - 1, xx] - gaussian_img[yy + 1, xx]
                    magnitude = math.sqrt(gx * gx + gy * gy)
                    orientation = math.degrees(math.atan2(gy, gx)) % 360
                    weight = math.exp(weight_factor * (dx * dx + dy * dy))
                    bin_idx = int(round(orientation / 10)) % 36
                    hist[bin_idx] += weight * magnitude

            max_val = hist.max()
            if max_val == 0:
                continue
            for bin_idx, value in enumerate(hist):
                if value >= 0.8 * max_val:
                    angle = (bin_idx * 10) % 360
                    oriented.append(
                        Keypoint(
                            x=kp.x,
                            y=kp.y,
                            octave=kp.octave,
                            layer=kp.layer,
                            sigma=kp.sigma,
                            orientation=math.radians(angle),
                        )
                    )

        return oriented

    # ----------------------- Descriptor computation ----------------------

    def _compute_descriptors(
        self, keypoints: List[Keypoint], gaussian_pyramid: List[List[np.ndarray]]
    ) -> np.ndarray:
        descriptors: List[np.ndarray] = []
        for kp in keypoints:
            octave_img = gaussian_pyramid[kp.octave][kp.layer]
            kp_scale = kp.sigma
            cos_o = math.cos(kp.orientation)
            sin_o = math.sin(kp.orientation)
            rows, cols = octave_img.shape

            descriptor = np.zeros((4, 4, 8), dtype=np.float32)
            window_size = int(round(8 * kp_scale))
            half_width = window_size // 2

            base_x = kp.x / (2**kp.octave)
            base_y = kp.y / (2**kp.octave)

            for dy in range(-half_width, half_width):
                for dx in range(-half_width, half_width):
                    # Rotate relative coordinates
                    rx = (cos_o * dx - sin_o * dy) + base_x
                    ry = (sin_o * dx + cos_o * dy) + base_y
                    ix, iy = int(round(rx)), int(round(ry))
                    if iy <= 0 or iy >= rows - 1 or ix <= 0 or ix >= cols - 1:
                        continue
                    gx = octave_img[iy, ix + 1] - octave_img[iy, ix - 1]
                    gy = octave_img[iy - 1, ix] - octave_img[iy + 1, ix]
                    magnitude = math.sqrt(gx * gx + gy * gy)
                    theta = (math.degrees(math.atan2(gy, gx)) - math.degrees(kp.orientation)) % 360

                    weight = math.exp(-((dx**2 + dy**2) / (2 * (0.5 * window_size) ** 2)))
                    magnitude *= weight

                    cell_x = int(
                        math.floor(
                            ((cos_o * dx - sin_o * dy) + half_width) / (half_width / 2 + 1e-5)
                        )
                    )
                    cell_y = int(
                        math.floor(
                            ((sin_o * dx + cos_o * dy) + half_width) / (half_width / 2 + 1e-5)
                        )
                    )
                    if cell_x < 0 or cell_x >= 4 or cell_y < 0 or cell_y >= 4:
                        continue
                    bin_idx = int(round(theta / 45)) % 8
                    descriptor[cell_y, cell_x, bin_idx] += magnitude

            vec = descriptor.ravel()
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
                vec = np.clip(vec, 0, 0.2)
                vec = vec / (np.linalg.norm(vec) + 1e-6)
            descriptors.append(vec)

        if not descriptors:
            return np.zeros((0, 128), dtype=np.float32)
        return np.vstack(descriptors)


# ---------------------------------------------------------------------------
# Matching + RANSAC helpers


def match_descriptors(
    desc_a: np.ndarray, desc_b: np.ndarray, ratio: float
) -> List[Match]:
    matches: List[Match] = []
    if desc_a.size == 0 or desc_b.size == 0:
        return matches
    for idx_a, vector in enumerate(desc_a):
        distances = np.linalg.norm(desc_b - vector, axis=1)
        if len(distances) < 2:
            continue
        best_idx = np.argmin(distances)
        best = distances[best_idx]
        distances[best_idx] = np.inf
        second = np.min(distances)
        if best < ratio * second:
            matches.append(Match(idx_a=idx_a, idx_b=int(best_idx), distance=float(best)))
    return matches


def compute_homography(pairs: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    A = []
    for src, dst in pairs:
        x, y = src
        u, v = dst
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    h = vt[-1, :]
    H = h.reshape(3, 3)
    return H / H[2, 2]


def ransac_homography(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    matches: List[Match] | List[cv2.DMatch],
    iterations: int,
    threshold: float,
) -> Tuple[np.ndarray | None, List[int]]:
    if len(matches) < 4:
        return None, []
    best_inliers: List[int] = []
    best_H: np.ndarray | None = None
    rng = random.Random(42)
    match_indices = list(range(len(matches)))

    for _ in range(iterations):
        sample_ids = rng.sample(match_indices, 4)
        pair_samples = []
        for idx in sample_ids:
            match = matches[idx]
            ia = match.queryIdx if hasattr(match, "queryIdx") else match.idx_a
            ib = match.trainIdx if hasattr(match, "trainIdx") else match.idx_b
            pair_samples.append((pts_a[ia], pts_b[ib]))
        H = compute_homography(pair_samples)

        inliers: List[int] = []
        for idx, match in enumerate(matches):
            ia = match.queryIdx if hasattr(match, "queryIdx") else match.idx_a
            ib = match.trainIdx if hasattr(match, "trainIdx") else match.idx_b
            pt_a = np.append(pts_a[ia], 1.0)
            projected = H @ pt_a
            projected /= projected[2]
            error = np.linalg.norm(projected[:2] - pts_b[ib])
            if error < threshold:
                inliers.append(idx)

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    return best_H, best_inliers


def draw_matches(
    img_a: np.ndarray,
    img_b: np.ndarray,
    keypoints_a: Iterable[Tuple[float, float]],
    keypoints_b: Iterable[Tuple[float, float]],
    matches: List[Match] | List[cv2.DMatch],
    inlier_indices: List[int],
) -> np.ndarray:
    kp_a = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints_a]
    kp_b = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints_b]
    if matches and isinstance(matches[0], Match):
        cv_matches = [
            cv2.DMatch(_queryIdx=m.idx_a, _trainIdx=m.idx_b, _distance=m.distance)
            for m in matches
        ]
    else:
        cv_matches = matches  # already cv2.DMatch
    inlier_matches = [cv_matches[idx] for idx in inlier_indices]
    vis = cv2.drawMatches(
        img_a,
        kp_a,
        img_b,
        kp_b,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return vis


# ---------------------------------------------------------------------------
# Script entry point


def keypoints_to_array(kps: Sequence[Keypoint]) -> np.ndarray:
    return np.array([[kp.x, kp.y] for kp in kps], dtype=np.float32)


def run_task(args: argparse.Namespace) -> None:
    img_a = load_image(args.image_a, args.resize_width)
    img_b = load_image(args.image_b, args.resize_width)
    gray_a = to_grayscale_float(img_a)
    gray_b = to_grayscale_float(img_b)

    siftr = SIFTFromScratch(
        num_octaves=args.octaves,
        num_scales=args.scales,
        sigma=args.sigma,
        contrast_threshold=args.contrast_threshold,
        edge_threshold=args.edge_threshold,
    )

    print("[Task2] Running custom SIFT pipeline ...")
    custom_kp_a, custom_desc_a = siftr.detect_and_compute(gray_a)
    custom_kp_b, custom_desc_b = siftr.detect_and_compute(gray_b)
    print(
        f"[Task2] Custom keypoints: image A={len(custom_kp_a)}, image B={len(custom_kp_b)}"
    )

    custom_matches = match_descriptors(custom_desc_a, custom_desc_b, args.ratio_test)
    print(f"[Task2] Custom matches before RANSAC: {len(custom_matches)}")
    custom_pts_a = keypoints_to_array(custom_kp_a)
    custom_pts_b = keypoints_to_array(custom_kp_b)
    custom_H, custom_inliers = ransac_homography(
        custom_pts_a, custom_pts_b, custom_matches, args.ransac_iters, args.ransac_threshold
    )
    print(f"[Task2] Custom RANSAC inliers: {len(custom_inliers)}")

    print("[Task2] Running OpenCV SIFT baseline ...")
    reference = cv2.SIFT_create()
    ref_kp_a, ref_desc_a = reference.detectAndCompute((gray_a * 255).astype(np.uint8), None)
    ref_kp_b, ref_desc_b = reference.detectAndCompute((gray_b * 255).astype(np.uint8), None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    ref_matches_knn = bf.knnMatch(ref_desc_a, ref_desc_b, k=2)
    ref_matches = []
    for m, n in ref_matches_knn:
        if m.distance < args.ratio_test * n.distance:
            ref_matches.append(m)
    ref_pts_a = np.array([kp.pt for kp in ref_kp_a], dtype=np.float32)
    ref_pts_b = np.array([kp.pt for kp in ref_kp_b], dtype=np.float32)
    ref_H, ref_inliers = ransac_homography(
        ref_pts_a, ref_pts_b, ref_matches, args.ransac_iters, args.ransac_threshold
    )
    print(f"[Task2] OpenCV keypoints: image A={len(ref_kp_a)}, image B={len(ref_kp_b)}")
    print(f"[Task2] OpenCV matches before RANSAC: {len(ref_matches)}")
    print(f"[Task2] OpenCV RANSAC inliers: {len(ref_inliers)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if custom_inliers:
        vis_custom = draw_matches(
            img_a,
            img_b,
            [(kp.x, kp.y) for kp in custom_kp_a],
            [(kp.x, kp.y) for kp in custom_kp_b],
            custom_matches,
            custom_inliers[:80],
        )
        cv2.imwrite(str(args.output_dir / "custom_sift_matches.jpg"), vis_custom)
    if ref_inliers:
        vis_ref = draw_matches(
            img_a,
            img_b,
            [kp.pt for kp in ref_kp_a],
            [kp.pt for kp in ref_kp_b],
            ref_matches,
            ref_inliers[:80],
        )
        cv2.imwrite(str(args.output_dir / "opencv_sift_matches.jpg"), vis_ref)

    summary = {
        "custom_keypoints_A": len(custom_kp_a),
        "custom_keypoints_B": len(custom_kp_b),
        "custom_matches": len(custom_matches),
        "custom_inliers": len(custom_inliers),
        "opencv_keypoints_A": len(ref_kp_a),
        "opencv_keypoints_B": len(ref_kp_b),
        "opencv_matches": len(ref_matches),
        "opencv_inliers": len(ref_inliers),
        "custom_homography": custom_H.tolist() if custom_H is not None else None,
        "opencv_homography": ref_H.tolist() if ref_H is not None else None,
    }
    summary_path = args.output_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"[Task2] Artefacts written to {args.output_dir.resolve()}")


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    run_task(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


