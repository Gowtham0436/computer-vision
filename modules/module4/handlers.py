"""
Module 4: Business logic handlers
Assignment 4 Implementation:
1. Image Stitching (using task1_stitch.py logic)
2. SIFT Feature Extraction (using task2_sift.py logic) with RANSAC
"""

import cv2
import numpy as np
import math
import random
import dataclasses
from typing import List, Tuple, Sequence, Iterable
from core.utils import decode_base64_image, encode_image_to_base64

# ---------------------------------------------------------------------------
# Data structures for SIFT
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Keypoint:
    """Minimal representation of a SIFT keypoint."""
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
# Image Stitching (from task1_stitch.py)
# ---------------------------------------------------------------------------

def create_stitcher() -> cv2.Stitcher:
    """Create OpenCV Stitcher instance."""
    mode = cv2.Stitcher_PANORAMA
    if hasattr(cv2, "Stitcher_create"):
        stitcher = cv2.Stitcher_create(mode)
    elif hasattr(cv2, "createStitcher"):
        stitcher = cv2.createStitcher(mode)
    else:
        raise RuntimeError("This version of OpenCV does not expose the Stitcher API.")
    return stitcher

def stitch_images_opencv(
    images: Sequence[np.ndarray],
) -> Tuple[bool, np.ndarray | None]:
    """Stitch images using OpenCV Stitcher API."""
    if len(images) < 2:
        return False, None
    
    try:
        stitcher = create_stitcher()
        status, panorama = stitcher.stitch(images)
        if status == cv2.Stitcher_OK:
            return True, panorama
        else:
            return False, None
    except Exception as e:
        print(f"Stitching error: {e}")
        return False, None

def make_side_by_side_comparison(
    stitched: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """
    Create side-by-side comparison of stitched panorama and reference image.
    From task1_stitch.py make_side_by_side function.
    """
    # Normalize heights for a fair comparison
    target_height = min(stitched.shape[0], reference.shape[0])
    
    def resize_to_height(image: np.ndarray) -> np.ndarray:
        scale = target_height / image.shape[0]
        if scale == 1.0:
            return image
        new_size = (int(image.shape[1] * scale), target_height)
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    
    stitched_resized = resize_to_height(stitched)
    ref_resized = resize_to_height(reference)
    
    padding = 20
    pad = np.full((target_height, padding, 3), 255, dtype=np.uint8)
    comparison = np.hstack([ref_resized, pad, stitched_resized])
    
    return comparison

def stitch_images_with_reference_handler(images_data, reference_image_data=None):
    """
    Combined Image Stitching and SIFT Feature Extraction
    
    Stitches multiple images together to create a panorama using OpenCV Stitcher,
    and automatically performs SIFT feature extraction and matching between consecutive images.
    Optionally compares with a reference image (e.g., from mobile phone).
    
    Args:
        images_data: List of base64 encoded images
        reference_image_data: Optional base64 encoded reference panorama image
        
    Returns:
        Dictionary with stitched image, SIFT results, reference comparison, and info
    """
    if not images_data or len(images_data) < 2:
        return {'success': False, 'error': 'Need at least 2 images for stitching'}
    
    # Decode all images
    images = []
    for img_data in images_data:
        img = decode_base64_image(img_data)
        if img is None:
            return {'success': False, 'error': 'Invalid image data'}
        images.append(img)
    
    if len(images) < 2:
        return {'success': False, 'error': 'Need at least 2 valid images'}
    
    # Resize images if too large (to avoid memory issues)
    max_width = 1800
    resize_width = 960  # For SIFT processing
    resized_images = []
    resized_for_sift = []
    for img in images:
        # Resize for stitching
        if img.shape[1] > max_width:
            scale = max_width / img.shape[1]
            new_size = (max_width, int(img.shape[0] * scale))
            resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            resized_img = img.copy()
        resized_images.append(resized_img)
        
        # Resize for SIFT (smaller for faster processing)
        if img.shape[1] > resize_width:
            scale = resize_width / img.shape[1]
            new_size = (resize_width, int(img.shape[0] * scale))
            sift_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            sift_img = img.copy()
        resized_for_sift.append(sift_img)
    
    # Use OpenCV Stitcher
    opencv_success, stitched_opencv = stitch_images_opencv(resized_images)
    
    result = {
        'success': opencv_success,
        'opencv_success': opencv_success,
        'num_images': len(images),
    }
    
    # SIFT Feature Extraction and Matching (automatically run on consecutive pairs)
    sift_results = []
    sift_matches_images = []
    sift_opencv_matches_images = []
    
    if len(resized_for_sift) >= 2:
        # Initialize SIFT detector
        siftr = SIFTFromScratch(
            num_octaves=4,
            num_scales=3,
            sigma=1.6,
            contrast_threshold=0.04,
            edge_threshold=10.0,
        )
        
        # Process each consecutive pair
        for i in range(len(resized_for_sift) - 1):
            img_a = resized_for_sift[i]
            img_b = resized_for_sift[i + 1]
            
            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if len(img_a.shape) == 3 else img_a
            gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if len(img_b.shape) == 3 else img_b
            gray_a_float = gray_a.astype(np.float32) / 255.0
            gray_b_float = gray_b.astype(np.float32) / 255.0
            
            try:
                # Custom SIFT
                custom_kp_a, custom_desc_a = siftr.detect_and_compute(gray_a_float)
                custom_kp_b, custom_desc_b = siftr.detect_and_compute(gray_b_float)
                
                # Match descriptors
                custom_matches = match_descriptors(custom_desc_a, custom_desc_b, ratio=0.75)
                custom_pts_a = keypoints_to_array(custom_kp_a)
                custom_pts_b = keypoints_to_array(custom_kp_b)
                
                # RANSAC
                custom_H, custom_inliers = ransac_homography(
                    custom_pts_a, custom_pts_b, custom_matches,
                    iterations=2000, threshold=3.0
                )
                
                # Draw matches - show all matches if no inliers, otherwise show inliers
                if custom_matches:
                    if custom_inliers:
                        vis_custom = draw_matches(
                            img_a, img_b,
                            [(kp.x, kp.y) for kp in custom_kp_a],
                            [(kp.x, kp.y) for kp in custom_kp_b],
                            custom_matches,
                            custom_inliers[:80],
                        )
                    else:
                        # Show all matches if no inliers found
                        vis_custom = draw_matches(
                            img_a, img_b,
                            [(kp.x, kp.y) for kp in custom_kp_a],
                            [(kp.x, kp.y) for kp in custom_kp_b],
                            custom_matches,
                            list(range(min(80, len(custom_matches)))),
                        )
                    sift_matches_images.append(encode_image_to_base64(vis_custom))
                else:
                    sift_matches_images.append(None)
                
                # OpenCV SIFT for comparison
                opencv_match_image = None
                opencv_inliers = 0
                try:
                    reference = cv2.SIFT_create()
                    ref_kp_a, ref_desc_a = reference.detectAndCompute((gray_a_float * 255).astype(np.uint8), None)
                    ref_kp_b, ref_desc_b = reference.detectAndCompute((gray_b_float * 255).astype(np.uint8), None)
                    
                    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                    ref_matches_knn = bf.knnMatch(ref_desc_a, ref_desc_b, k=2)
                    ref_matches = []
                    for m, n in ref_matches_knn:
                        if m.distance < 0.75 * n.distance:
                            ref_matches.append(m)
                    
                    ref_pts_a = np.array([kp.pt for kp in ref_kp_a], dtype=np.float32)
                    ref_pts_b = np.array([kp.pt for kp in ref_kp_b], dtype=np.float32)
                    ref_H, ref_inliers = ransac_homography(
                        ref_pts_a, ref_pts_b, ref_matches,
                        iterations=2000, threshold=3.0
                    )
                    
                    opencv_inliers = len(ref_inliers) if ref_inliers else 0
                    
                    # Draw OpenCV matches
                    if ref_matches:
                        if ref_inliers:
                            vis_opencv = draw_matches(
                                img_a, img_b,
                                [kp.pt for kp in ref_kp_a],
                                [kp.pt for kp in ref_kp_b],
                                ref_matches,
                                ref_inliers[:80],
                            )
                        else:
                            vis_opencv = draw_matches(
                                img_a, img_b,
                                [kp.pt for kp in ref_kp_a],
                                [kp.pt for kp in ref_kp_b],
                                ref_matches,
                                list(range(min(80, len(ref_matches)))),
                            )
                        opencv_match_image = encode_image_to_base64(vis_opencv)
                except Exception as e:
                    opencv_inliers = 0
                    opencv_match_image = None
                
                sift_results.append({
                    'pair': f'{i+1}-{i+2}',
                    'custom_keypoints_a': len(custom_kp_a),
                    'custom_keypoints_b': len(custom_kp_b),
                    'custom_matches': len(custom_matches),
                    'custom_inliers': len(custom_inliers) if custom_inliers else 0,
                    'opencv_inliers': opencv_inliers,
                    'homography_found': custom_H is not None
                })
                sift_opencv_matches_images.append(opencv_match_image)
            except Exception as e:
                sift_results.append({
                    'pair': f'{i+1}-{i+2}',
                    'error': str(e)
                })
                sift_matches_images.append(None)
                sift_opencv_matches_images.append(None)
    
    result['sift_results'] = sift_results
    result['sift_matches_images'] = sift_matches_images
    result['sift_opencv_matches_images'] = sift_opencv_matches_images
    
    if opencv_success and stitched_opencv is not None:
        stitched_opencv_b64 = encode_image_to_base64(stitched_opencv)
        result['stitched_opencv'] = stitched_opencv_b64
        result['comparison_note'] = f"Successfully stitched {len(images)} images. SIFT features extracted from {len(sift_results)} image pair(s)."
        
        # If reference image provided, include it separately for manual comparison
        if reference_image_data:
            try:
                reference_img = decode_base64_image(reference_image_data)
                if reference_img is not None:
                    result['reference_image'] = encode_image_to_base64(reference_img)
                    result['has_reference'] = True
                else:
                    result['has_reference'] = False
                    result['reference_error'] = 'Invalid reference image'
            except Exception as e:
                result['has_reference'] = False
                result['reference_error'] = str(e)
        else:
            result['has_reference'] = False
    else:
        result['stitched_opencv'] = None
        result['comparison_note'] = "Stitching failed. Try ensuring images have 30-50% overlap and sufficient features."
        result['has_reference'] = False
    
    return result

# ---------------------------------------------------------------------------
# SIFT Implementation (from task2_sift.py)
# ---------------------------------------------------------------------------

class SIFTFromScratch:
    """Custom SIFT implementation from scratch."""
    
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
# Matching + RANSAC helpers (from task2_sift.py)
# ---------------------------------------------------------------------------

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

def keypoints_to_array(kps: Sequence[Keypoint]) -> np.ndarray:
    return np.array([[kp.x, kp.y] for kp in kps], dtype=np.float32)

def keypoints_to_cv_keypoints(kps: Sequence[Keypoint]) -> List[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(kp.x), float(kp.y), 1) for kp in kps]

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
        cv_matches = matches
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
# Handler functions for web interface
# ---------------------------------------------------------------------------

def extract_sift_features_handler(image1_data, image2_data=None, compare_with_opencv=True):
    """
    Problem 2: SIFT Feature Extraction from Scratch
    
    Implements SIFT feature extraction from scratch and compares with OpenCV SIFT.
    If two images are provided, also performs matching with RANSAC.
    
    Args:
        image1_data: Base64 encoded first image
        image2_data: Base64 encoded second image (optional, for matching)
        compare_with_opencv: Whether to compare with OpenCV SIFT
        
    Returns:
        Dictionary with SIFT features, keypoints, and comparison
    """
    img1 = decode_base64_image(image1_data)
    if img1 is None:
        return {'success': False, 'error': 'Invalid image'}
    
    # Resize if too large
    resize_width = 960
    if img1.shape[1] > resize_width:
        scale = resize_width / img1.shape[1]
        new_size = (resize_width, int(img1.shape[0] * scale))
        img1 = cv2.resize(img1, new_size, interpolation=cv2.INTER_AREA)
    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray1_float = gray1.astype(np.float32) / 255.0
    
    # Custom SIFT
    siftr = SIFTFromScratch(
        num_octaves=4,
        num_scales=3,
        sigma=1.6,
        contrast_threshold=0.04,
        edge_threshold=10.0,
    )
    
    custom_kp_a, custom_desc_a = siftr.detect_and_compute(gray1_float)
    
    # Draw custom keypoints
    img_with_custom_kp = img1.copy()
    custom_cv_kp = keypoints_to_cv_keypoints(custom_kp_a)
    cv2.drawKeypoints(img1, custom_cv_kp, img_with_custom_kp, 
                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    result = {
        'success': True,
        'original_image': encode_image_to_base64(img1),
        'custom_keypoints': encode_image_to_base64(img_with_custom_kp),
        'custom_kp_count': len(custom_kp_a),
    }
    
    # OpenCV SIFT comparison
    if compare_with_opencv:
        try:
            reference = cv2.SIFT_create()
            ref_kp_a, ref_desc_a = reference.detectAndCompute((gray1_float * 255).astype(np.uint8), None)
            
            img_with_opencv_kp = img1.copy()
            cv2.drawKeypoints(img1, ref_kp_a, img_with_opencv_kp,
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
    comparison = np.hstack([img_with_custom_kp, img_with_opencv_kp])
    
            result.update({
        'opencv_keypoints': encode_image_to_base64(img_with_opencv_kp),
        'comparison': encode_image_to_base64(comparison),
                'opencv_kp_count': len(ref_kp_a),
                'opencv_success': True,
            })
        except Exception as e:
            result.update({
                'opencv_success': False,
                'opencv_error': str(e),
            })
    
    # If two images provided, do matching
    if image2_data:
    img2 = decode_base64_image(image2_data)
        if img2 is None:
            return {'success': False, 'error': 'Invalid second image'}
        
        if img2.shape[1] > resize_width:
            scale = resize_width / img2.shape[1]
            new_size = (resize_width, int(img2.shape[0] * scale))
            img2 = cv2.resize(img2, new_size, interpolation=cv2.INTER_AREA)
        
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
        gray2_float = gray2.astype(np.float32) / 255.0
        
        # Custom SIFT on second image
        custom_kp_b, custom_desc_b = siftr.detect_and_compute(gray2_float)
        
        # Match descriptors
        custom_matches = match_descriptors(custom_desc_a, custom_desc_b, ratio=0.75)
        custom_pts_a = keypoints_to_array(custom_kp_a)
        custom_pts_b = keypoints_to_array(custom_kp_b)
    
        # RANSAC
        custom_H, custom_inliers = ransac_homography(
            custom_pts_a, custom_pts_b, custom_matches, 
            iterations=2000, threshold=3.0
        )
        
        # Draw matches
        if custom_inliers:
            vis_custom = draw_matches(
                img1, img2,
                [(kp.x, kp.y) for kp in custom_kp_a],
                [(kp.x, kp.y) for kp in custom_kp_b],
                custom_matches,
                custom_inliers[:80],
            )
            result['custom_matches'] = encode_image_to_base64(vis_custom)
            result['custom_matches_count'] = len(custom_matches)
            result['custom_inliers_count'] = len(custom_inliers)
            result['custom_homography_found'] = custom_H is not None
        
        # OpenCV SIFT matching
        if compare_with_opencv and 'opencv_success' in result and result['opencv_success']:
            try:
                reference = cv2.SIFT_create()
                ref_kp_a, ref_desc_a = reference.detectAndCompute((gray1_float * 255).astype(np.uint8), None)
                ref_kp_b, ref_desc_b = reference.detectAndCompute((gray2_float * 255).astype(np.uint8), None)
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
                ref_matches_knn = bf.knnMatch(ref_desc_a, ref_desc_b, k=2)
                ref_matches = []
                for m, n in ref_matches_knn:
                    if m.distance < 0.75 * n.distance:
                        ref_matches.append(m)
                
                ref_pts_a = np.array([kp.pt for kp in ref_kp_a], dtype=np.float32)
                ref_pts_b = np.array([kp.pt for kp in ref_kp_b], dtype=np.float32)
                ref_H, ref_inliers = ransac_homography(
                    ref_pts_a, ref_pts_b, ref_matches, 
                    iterations=2000, threshold=3.0
                )
                
                if ref_inliers:
                    vis_ref = draw_matches(
                        img1, img2,
                        [kp.pt for kp in ref_kp_a],
                        [kp.pt for kp in ref_kp_b],
                        ref_matches,
                        ref_inliers[:80],
                    )
                    result['opencv_matches'] = encode_image_to_base64(vis_ref)
                    result['opencv_matches_count'] = len(ref_matches)
                    result['opencv_inliers_count'] = len(ref_inliers)
                    result['opencv_homography_found'] = ref_H is not None
            except Exception as e:
                result['opencv_matching_error'] = str(e)
    
    return result
