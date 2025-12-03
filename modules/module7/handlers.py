"""
Module 7: Stereo Calibration, Pose Estimation, and Hand Tracking

New implementation using the provided classes. Existing handler
functions are kept so the rest of the app (routes/templates) work
without changes.
"""

import os
import csv
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

from core.utils import decode_base64_image, encode_image_to_base64


class StereoCalibration:
    """Stereo camera calibration and depth estimation"""

    def __init__(self):
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        self.baseline = None

    def calibrate_stereo(self, left_images, right_images, pattern_size=(9, 6), square_size=1.0):
        """
        Calibrate stereo camera pair.
        This method is kept for completeness but the web handler uses
        the compute_disparity/measure_object_size path below.
        """
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        objpoints = []       # 3D points in real world
        imgpoints_left = []  # 2D points in left image
        imgpoints_right = []  # 2D points in right image

        for left_img, right_img in zip(left_images, right_images):
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size)

            if ret_left and ret_right:
                objpoints.append(objp)

                corners_left = cv2.cornerSubPix(
                    gray_left,
                    corners_left,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                corners_right = cv2.cornerSubPix(
                    gray_right,
                    corners_right,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )

                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)

        if len(objpoints) == 0:
            return False

        img_shape = left_images[0].shape[:2][::-1]

        _, self.camera_matrix_left, self.dist_coeffs_left, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_left, img_shape, None, None
        )

        _, self.camera_matrix_right, self.dist_coeffs_right, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_right, img_shape, None, None
        )

        flags = cv2.CALIB_FIX_INTRINSIC
        ret, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.camera_matrix_right,
            self.dist_coeffs_right,
            img_shape,
            flags=flags,
        )

        self.baseline = np.linalg.norm(self.T)
        return ret

    def compute_disparity(self, left_image, right_image):
        """Compute disparity map from stereo pair"""
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM_create(numDisparities=16 * 5, blockSize=15)
        disparity = stereo.compute(gray_left, gray_right)

        disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return disparity, disparity_vis

    def estimate_depth(self, disparity, focal_length=None):
        """
        Estimate depth from disparity.
        Depth = (Focal Length * Baseline) / Disparity
        """
        if focal_length is None and self.camera_matrix_left is not None:
            focal_length = self.camera_matrix_left[0, 0]

        if focal_length is None or self.baseline is None:
            return None

        depth = np.zeros_like(disparity, dtype=np.float32)
        valid_disparity = disparity > 0
        depth[valid_disparity] = (focal_length * self.baseline) / disparity[valid_disparity]
        return depth

    def measure_object_size(self, left_image, right_image, bbox):
        """
        Measure object size using stereo reconstruction.
        Returns object dimensions (width, height, depth).
        """
        disparity, _ = self.compute_disparity(left_image, right_image)
        depth_map = self.estimate_depth(disparity)

        if depth_map is None:
            return None

        x, y, w, h = bbox
        roi_depth = depth_map[y : y + h, x : x + w]
        roi_valid = roi_depth[roi_depth > 0]
        if roi_valid.size == 0:
            return None

        avg_depth = float(np.median(roi_valid))
        if np.isnan(avg_depth) or avg_depth <= 0:
            return None

        focal_length = (
            self.camera_matrix_left[0, 0] if self.camera_matrix_left is not None else 700.0
        )

        real_width = (w * avg_depth) / focal_length
        real_height = (h * avg_depth) / focal_length

        return {
            "width": float(real_width),
            "height": float(real_height),
            "depth": avg_depth,
            "bbox": bbox,
        }


class PoseEstimation:
    """Real-time pose estimation using Mediapipe"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.csv_data = []
        self.frame_count = 0

    def process_frame(self, frame):
        """Process frame for pose estimation."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        annotated_frame = frame.copy()
        pose_data = None

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )

            pose_data = self.extract_pose_data(results.pose_landmarks, self.frame_count)
            self.csv_data.append(pose_data)
            self.frame_count += 1

        return annotated_frame, results.pose_landmarks, pose_data

    def extract_pose_data(self, landmarks, frame_num):
        """Extract pose data for logging."""
        data = {"frame": frame_num}
        for idx, landmark in enumerate(landmarks.landmark):
            landmark_name = self.mp_pose.PoseLandmark(idx).name
            data[f"{landmark_name}_x"] = landmark.x
            data[f"{landmark_name}_y"] = landmark.y
            data[f"{landmark_name}_z"] = landmark.z
            data[f"{landmark_name}_visibility"] = landmark.visibility
        return data

    def save_to_csv(self, filename):
        """Save collected pose data to CSV file."""
        if not self.csv_data:
            return False

        keys = self.csv_data[0].keys()
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.csv_data)
        return True

    def reset_data(self):
        self.csv_data = []
        self.frame_count = 0


class HandTracking:
    """Real-time hand tracking using Mediapipe"""

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2
        )
        self.csv_data = []
        self.frame_count = 0

    def process_frame(self, frame):
        """Process frame for hand tracking."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        annotated_frame = frame.copy()
        hand_data = None

        if results.multi_hand_landmarks:
            hand_data = {"frame": self.frame_count, "num_hands": len(results.multi_hand_landmarks)}

            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                if results.multi_handedness:
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                    hand_data[f"hand_{hand_idx}_type"] = handedness

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmark_name = self.mp_hands.HandLandmark(idx).name
                    hand_data[f"hand_{hand_idx}_{landmark_name}_x"] = landmark.x
                    hand_data[f"hand_{hand_idx}_{landmark_name}_y"] = landmark.y
                    hand_data[f"hand_{hand_idx}_{landmark_name}_z"] = landmark.z

            self.csv_data.append(hand_data)
            self.frame_count += 1

        return annotated_frame, getattr(results, "multi_hand_landmarks", None), hand_data

    def save_to_csv(self, filename):
        """Save collected hand data to CSV file."""
        if not self.csv_data:
            return False

        all_keys = set()
        for data in self.csv_data:
            all_keys.update(data.keys())
        keys = sorted(all_keys)

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.csv_data)
        return True

    def reset_data(self):
        self.csv_data = []
        self.frame_count = 0


# ----------------------------------------------------------------------
# Global instances used by the Flask handlers
# ----------------------------------------------------------------------

_stereo = StereoCalibration()
_pose = PoseEstimation()
_hands = HandTracking()


def estimate_size_calibrated_stereo_handler(left_image_data, right_image_data, camera_params, object_type="rectangular"):
    """
    Problem 1: Calibrated Stereo Size Estimation using the new StereoCalibration class.
    """
    left_img = decode_base64_image(left_image_data)
    right_img = decode_base64_image(right_image_data)
    
    if left_img is None or right_img is None:
        return {"success": False, "error": "Invalid image(s)"}
    
    # Build a simple intrinsic matrix from provided camera params
    fx = float(camera_params.get("fx", 1000.0))
    fy = float(camera_params.get("fy", fx))
    cx = float(camera_params.get("cx", left_img.shape[1] / 2))
    cy = float(camera_params.get("cy", left_img.shape[0] / 2))
    baseline = float(camera_params.get("baseline", 0.1))

    _stereo.camera_matrix_left = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    _stereo.baseline = baseline

    disparity, disparity_vis = _stereo.compute_disparity(left_img, right_img)

    h, w = left_img.shape[:2]
    roi_size = min(w, h) // 4
    center_x, center_y = w // 2, h // 2
    x = max(0, center_x - roi_size)
    y = max(0, center_y - roi_size)
    w_roi = min(roi_size * 2, w - x)
    h_roi = min(roi_size * 2, h - y)
    bbox = (x, y, w_roi, h_roi)

    size_info = _stereo.measure_object_size(left_img, right_img, bbox)
    if size_info is None:
        return {
            "success": False,
            "error": "Unable to estimate object size from disparity. Check stereo alignment.",
        }

    # Draw ROI on left image for visualization
        annotated = left_img.copy()
    cv2.rectangle(annotated, (x, y), (x + w_roi, y + h_roi), (0, 255, 0), 2)
    cv2.putText(
        annotated,
        f"Z: {size_info['depth']:.3f}m",
        (x, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
    
    return {
        "success": True,
        "left_image": encode_image_to_base64(left_img),
        "right_image": encode_image_to_base64(right_img),
        "disparity_map": encode_image_to_base64(disparity_color),
        "annotated_image": encode_image_to_base64(annotated),
        "distance_z_m": float(size_info["depth"]),
        "object_type": object_type,
        "dimensions": {
            "width_m": float(size_info["width"]),
            "height_m": float(size_info["height"]),
        },
        "formula": "Depth = (focal_length * baseline) / disparity (via new StereoCalibration class)",
    }


def estimate_pose_hand_tracking_handler(image_data, use_mediapipe=True):
    """
    Problem 3: Real-time Pose Estimation and Hand Tracking using the new classes.
    """
    image = decode_base64_image(image_data)
    if image is None:
        return {"success": False, "error": "Invalid image"}

    try:
        # Pose first
        annotated_pose, pose_landmarks, pose_data = _pose.process_frame(image)
        # Then hands on top of pose drawing
        annotated_full, hand_landmarks_list, hand_data = _hands.process_frame(annotated_pose)

        pose_count = len(pose_landmarks.landmark) if pose_landmarks else 0
        hand_count = len(hand_landmarks_list) if hand_landmarks_list else 0

        # For simplicity we don't return the full CSV text; the caller can still
        # trigger save via save_pose_csv_to_file using accumulated data.
        return {
            "success": True,
            "original_image": encode_image_to_base64(image),
            "annotated_image": encode_image_to_base64(annotated_full),
            "num_pose_landmarks": int(pose_count),
            "num_hands": int(hand_count),
            "method": "MediaPipe (new classes)",
        }
    except Exception as e:
        return {"success": False, "error": f"MediaPipe processing failed: {e}"}


def save_pose_csv_to_file(csv_data=None, filename=None):
    """
    Save collected pose/hand data to CSV.

    For backwards compatibility with existing routes that pass a CSV
    string, we still accept `csv_data` but if it is None we instead
    dump the accumulated data from the PoseEstimation / HandTracking
    objects.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/outputs/pose_data_{timestamp}.csv"
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    if csv_data is not None:
        # Old behavior: write CSV string directly.
        with open(filename, "w", newline="") as f:
            f.write(csv_data)
        return filename

    # New behavior: combine pose + hand CSV from internal buffers.
    combined = []
    combined.extend(_pose.csv_data)
    combined.extend(_hands.csv_data)
    if not combined:
        return filename

    keys = sorted({k for row in combined for k in row.keys()})
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(combined)
    
    return filename


