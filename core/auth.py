"""
Core authentication module using OpenCV DNN face recognition
Deployment-friendly: No dlib compilation required
High accuracy: Uses state-of-the-art deep learning models
"""

import os
import pickle
import base64
import numpy as np
import cv2
import requests

class FaceAuthenticator:
    """Face authentication using OpenCV DNN (deployment-friendly, high accuracy)"""
    
    def __init__(self, encodings_file="face_encodings.pkl", models_dir="models"):
        self.encodings_file = encodings_file
        self.models_dir = models_dir
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Initialize models
        self.face_detector = None
        self.face_recognizer = None
        self._initialize_models()
        
        # Load saved encodings
        self.load_encodings()
    
    def _initialize_models(self):
        """Initialize face detection and recognition models"""
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Download and initialize face detector (Yunet - fast and accurate)
        detector_path = os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx")
        if not os.path.exists(detector_path):
            print("Downloading face detection model...")
            # Try multiple URLs (GitHub structure may vary)
            urls = [
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            ]
            self._download_model_with_fallback(urls, detector_path)
        
        # Initialize Yunet face detector
        self.face_detector = cv2.FaceDetectorYN.create(
            detector_path,
            "",
            (320, 320),  # Input size
            0.9,  # Score threshold
            0.3,  # NMS threshold
            5000  # Top K
        )
        
        # Download and initialize face recognizer (ResNet-based, 128D embeddings)
        recognizer_path = os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")
        if not os.path.exists(recognizer_path):
            print("Downloading face recognition model...")
            # Try multiple URLs
            urls = [
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                "https://github.com/opencv/opencv_zoo/raw/master/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            ]
            self._download_model_with_fallback(urls, recognizer_path)
        
        # Initialize face recognizer
        self.face_recognizer = cv2.FaceRecognizerSF.create(
            recognizer_path,
            ""
        )
        
        print("Face recognition models initialized successfully!")
    
    def _download_model_with_fallback(self, urls, save_path):
        """Download model file from URL with fallback options"""
        last_error = None
        for url in urls:
            try:
                print(f"Trying: {url}")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                if int(percent) % 20 == 0:  # Print every 20%
                                    print(f"  Progress: {percent:.1f}%")
                
                print(f"  Model downloaded successfully: {os.path.basename(save_path)}")
                return  # Success!
            except Exception as e:
                last_error = e
                print(f"  Failed: {e}")
                continue  # Try next URL
        
        # All URLs failed
        print(f"\nError: Could not download model from any URL")
        print(f"Please manually download one of these files:")
        for url in urls:
            print(f"  - {url}")
        print(f"Save to: {save_path}")
        raise Exception(f"Failed to download model: {last_error}")
    
    def _download_model(self, url, save_path):
        """Download model file from URL (legacy method)"""
        self._download_model_with_fallback([url], save_path)
    
    def _detect_faces(self, image):
        """
        Detect faces in image using Yunet detector
        
        Returns:
            List of face detection arrays (each face has 15 values: x, y, w, h, landmarks, score)
        """
        height, width = image.shape[:2]
        self.face_detector.setInputSize((width, height))
        
        # Detect faces
        _, faces = self.face_detector.detect(image)
        
        if faces is None:
            return []
        
        # Return full face detection arrays (needed for alignCrop)
        return faces.tolist() if isinstance(faces, np.ndarray) else faces
    
    def _get_face_encoding(self, image, face_detection):
        """
        Extract face encoding (128D feature vector) from detected face
        
        Args:
            image: Input image (BGR)
            face_detection: Face detection array from Yunet (15 values)
            
        Returns:
            numpy array: 128-dimensional face encoding
        """
        # Convert to numpy array if needed
        if isinstance(face_detection, list):
            face_detection = np.array(face_detection, dtype=np.float32)
        
        # Align face for better recognition accuracy
        face_align = self.face_recognizer.alignCrop(image, face_detection)
        
        # Extract face features (128D vector)
        face_feature = self.face_recognizer.feature(face_align)
        
        # Normalize the feature vector
        face_feature = face_feature / np.linalg.norm(face_feature)
        
        return face_feature.flatten()
    
    def load_encodings(self):
        """Load saved face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_encodings)} face encoding(s)")
            except Exception as e:
                print(f"Error loading encodings: {e}")
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
    
    def register_face_from_image(self, image_data, name="User"):
        """
        Register face from base64 image data
        
        Args:
            image_data: Base64 encoded image string
            name: Name to associate with the face
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            # Decode base64 image
            img_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return False, "Invalid image data"
            
            # Improve image quality
            frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            
            # Detect faces
            faces = self._detect_faces(frame)
            
            if not faces:
                return False, "No face detected in image. Please ensure your face is clearly visible."
            
            # Use the largest face (most likely the main subject)
            # Face detection format: [x, y, w, h, landmarks..., score]
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)  # Sort by area (w * h)
            main_face = np.array(faces[0], dtype=np.float32)
            
            # Extract face encoding
            face_encoding = self._get_face_encoding(frame, main_face)
            
            # Check if face already exists (prevent duplicates)
            if len(self.known_face_encodings) > 0:
                distances = [np.linalg.norm(face_encoding - known_enc) for known_enc in self.known_face_encodings]
                min_distance = min(distances) if distances else float('inf')
                
                # If very similar face exists, update it instead of adding duplicate
                if min_distance < 0.3:  # Threshold for same person
                    min_idx = np.argmin(distances)
                    self.known_face_encodings[min_idx] = face_encoding
                    self.known_face_names[min_idx] = name
                    self.save_encodings()
                    return True, f"Face updated for {name}!"
            
            # Add new face
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.save_encodings()
            return True, f"Face registered successfully for {name}!"
            
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def authenticate_from_image(self, image_data, confidence_threshold=0.6):
        """
        Authenticate from base64 image data
        
        Args:
            image_data: Base64 encoded image string
            confidence_threshold: Minimum cosine similarity for authentication (0-1)
                                 Higher = more strict (default 0.6 = 60% similarity)
            
        Returns:
            Tuple of (success: bool, name: str, confidence: float)
        """
        try:
            if not self.known_face_encodings:
                return False, "No registered faces found. Please register first.", 0
            
            # Decode base64 image
            img_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return False, "Invalid image data", 0
            
            # Detect faces
            faces = self._detect_faces(frame)
            
            if not faces:
                return False, "No face detected. Please position your face properly.", 0
            
            # Use the largest face
            # Face detection format: [x, y, w, h, landmarks..., score]
            faces.sort(key=lambda f: f[2] * f[3], reverse=True)  # Sort by area (w * h)
            main_face = np.array(faces[0], dtype=np.float32)
            
            # Extract face encoding
            face_encoding = self._get_face_encoding(frame, main_face)
            
            # Compare with known faces using cosine similarity
            # Higher similarity = better match (range: 0-1)
            similarities = []
            for known_enc in self.known_face_encodings:
                # Cosine similarity: dot product of normalized vectors
                similarity = np.dot(face_encoding, known_enc)
                similarities.append(similarity)
            
            if not similarities:
                return False, "No registered faces to compare", 0
            
            # Find best match
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            
            # Check if similarity meets threshold
            # Using cosine similarity: 0.6 = 60% match, 0.7 = 70% match, etc.
            if best_similarity >= confidence_threshold:
                name = self.known_face_names[best_match_index]
                confidence = best_similarity  # Already in 0-1 range
                return True, name, confidence
            else:
                return False, f"Authentication failed. Similarity: {best_similarity:.2%} (required: {confidence_threshold:.0%})", best_similarity
                
        except Exception as e:
            return False, f"Error: {str(e)}", 0

# Global instance
face_auth = FaceAuthenticator()
