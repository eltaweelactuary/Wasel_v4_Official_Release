"""
Wasel v4 Pro: Core Engine
YOLOv8-Pose for skeletal tracking + TensorFlow LSTM for temporal sign classification.
Replaces: MediaPipe + RandomForest from v3.
"""

import os
import cv2
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from .gcp_utils import GCSManager, setup_cloud_logging, is_running_on_gcp

# Conditional imports with graceful fallback
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

logger = logging.getLogger(__name__)


class WaselEngine:
    """
    Wasel v4 Pro: Unified Inference Engine.
    
    Architecture:
        Input Frame → YOLOv8-Pose (Keypoints) → Feature Buffer → TF LSTM (Classification) → Label
    
    Fallback:
        If YOLO unavailable → MediaPipe Holistic (v3 compat)
        If TF unavailable → RandomForest from pickle (v3 compat)
    """

    # ─── Feature Dimensions ───
    YOLO_BODY_KEYPOINTS = 17      # COCO keypoints (x, y, conf) = 51 features
    YOLO_FEATURES = 17 * 3        # 51
    MP_FEATURES = 225             # v3 compat: Hands(126) + Pose(99)
    
    def __init__(self, data_dir: str = "./wasel_v4_data",
                 yolo_weights: str = "yolov8n-pose.pt",
                 tf_model_path: Optional[str] = None,
                 legacy_model_path: Optional[str] = None):
        """
        Initialize the engine with smart backend selection.
        
        Args:
            data_dir: Directory for persisted assets (landmarks, models)
            yolo_weights: Path to YOLOv8-Pose weights
            tf_model_path: Path to trained TensorFlow .h5 model
            legacy_model_path: Path to v3 RandomForest .pkl (fallback)
        """
        self.data_dir = Path(data_dir)
        self.landmarks_dir = self.data_dir / "landmarks"
        self.models_dir = self.data_dir / "models"
        self.videos_dir = self.data_dir / "videos"
        
        for d in [self.landmarks_dir, self.models_dir, self.videos_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # ─── GCP Integration ───
        setup_cloud_logging()
        self.gcs = GCSManager()
        self._sync_with_cloud()
        
        # ─── Backend Selection ───
        self.backend = self._select_backend(yolo_weights, tf_model_path, legacy_model_path)
        
        # ─── State ───
        self.landmark_dict: Dict[str, np.ndarray] = {}
        self.classifier = None
        self.label_encoder = None
        self._load_cached_landmarks()
        self._load_classifier()
        
        logger.info(f"🚀 WaselEngine initialized | Backend: {self.backend['pose']} + {self.backend['classifier']}")
        logger.info(f"   Vocabulary: {len(self.landmark_dict)} words loaded")

    def _sync_with_cloud(self):
        """Sync local assets with GCS on initialization."""
        if self.gcs.client:
            logger.info("☁️ Syncing assets with Cloud Storage...")
            self.gcs.sync_directory(self.landmarks_dir, "landmarks/")
            self.gcs.sync_directory(self.models_dir, "models/")

    def _select_backend(self, yolo_weights, tf_model_path, legacy_model_path) -> dict:
        """Smart backend selection with graceful degradation."""
        backend = {"pose": "none", "classifier": "none", "pose_model": None, "clf_model": None}
        
        # Pose Backend
        if YOLO_AVAILABLE:
            try:
                # Ensure the model file exists, download if necessary
                yolo_path = self.models_dir / yolo_weights
                if not yolo_path.exists():
                    logger.info(f"⏳ Downloading YOLOv8 model to {yolo_path}...")
                    # YOLO() will automatically download from Ultralytics if not found locally
                    # By passing the absolute path, we force it to download there
                
                backend["pose_model"] = YOLO(str(yolo_path))
                backend["pose"] = "yolo"
                logger.info("✅ YOLOv8-Pose loaded")
            except Exception as e:
                logger.warning(f"⚠️ YOLO load failed: {e}")
        
        if backend["pose"] == "none" and MP_AVAILABLE:
            backend["pose"] = "mediapipe"
            logger.info("✅ Fallback: MediaPipe Holistic")
        
        # Classifier Backend
        if TF_AVAILABLE and tf_model_path and Path(tf_model_path).exists():
            try:
                backend["clf_model"] = tf.keras.models.load_model(tf_model_path)
                backend["classifier"] = "tensorflow"
                logger.info("✅ TensorFlow classifier loaded")
            except Exception as e:
                logger.warning(f"⚠️ TF model load failed: {e}")
        
        if backend["classifier"] == "none" and legacy_model_path and Path(legacy_model_path).exists():
            try:
                with open(legacy_model_path, 'rb') as f:
                    self.classifier, self.label_encoder = pickle.load(f)
                backend["classifier"] = "sklearn"
                logger.info("✅ Fallback: RandomForest from v3")
            except Exception as e:
                logger.warning(f"⚠️ Legacy model load failed: {e}")
        
        return backend

    # ═══════════════════════════════════════════
    # ─── POSE EXTRACTION ───
    # ═══════════════════════════════════════════

    def extract_keypoints_yolo(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract body keypoints using YOLOv8-Pose. Returns flattened (51,) array."""
        model = self.backend["pose_model"]
        if model is None:
            return None
        
        results = model(frame, verbose=False)
        if not results or not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return None
        
        kp = results[0].keypoints.data[0].cpu().numpy()  # (17, 3)
        return kp.flatten()

    def extract_keypoints_mediapipe(self, frame: np.ndarray, holistic) -> Optional[np.ndarray]:
        """Extract keypoints using MediaPipe Holistic (v3 compat). Returns (225,) array."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame
        results = holistic.process(rgb)
        
        ref_x, ref_y = 0.5, 0.5
        if results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]
            ref_x, ref_y = nose.x, nose.y
        
        def get_coords(attr, n=21):
            if not attr:
                return [0.0] * (n * 3)
            pts = attr.landmark[:n]
            return [c for lm in pts for c in [lm.x - ref_x, lm.y - ref_y, lm.z]]
        
        features = []
        features.extend(get_coords(results.left_hand_landmarks, 21))   # 63
        features.extend(get_coords(results.right_hand_landmarks, 21))  # 63
        features.extend(get_coords(results.pose_landmarks, 33))        # 99
        return np.array(features)  # (225,)

    def extract_keypoints(self, frame: np.ndarray, holistic=None) -> Optional[np.ndarray]:
        """Unified keypoint extraction — auto-selects best backend."""
        if self.backend["pose"] == "yolo":
            return self.extract_keypoints_yolo(frame)
        elif self.backend["pose"] == "mediapipe" and holistic:
            return self.extract_keypoints_mediapipe(frame, holistic)
        return None

    # ═══════════════════════════════════════════
    # ─── VIDEO PROCESSING ───
    # ═══════════════════════════════════════════

    def extract_from_video(self, video_path: str, max_frames: int = 120) -> Optional[np.ndarray]:
        """
        Extract keypoint sequence from video.
        Returns: np.ndarray of shape (N_frames, N_features)
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return None
        
        sequences = []
        holistic = None
        
        # Create MediaPipe if needed
        if self.backend["pose"] == "mediapipe" and MP_AVAILABLE:
            holistic = mp.solutions.holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        try:
            count = 0
            while cap.isOpened() and count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                kp = self.extract_keypoints(frame, holistic)
                if kp is not None:
                    sequences.append(kp)
                count += 1
        finally:
            cap.release()
            if holistic:
                holistic.close()
        
        if not sequences:
            return None
        return np.array(sequences)

    # ═══════════════════════════════════════════
    # ─── CLASSIFICATION ───
    # ═══════════════════════════════════════════

    def predict(self, sequence: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Predict sign from a keypoint sequence.
        Handles both TF LSTM and sklearn RandomForest.
        """
        if sequence is None or len(sequence) < 5:
            return None, 0.0
        
        if self.backend["classifier"] == "tensorflow":
            return self._predict_tf(sequence)
        elif self.backend["classifier"] == "sklearn":
            return self._predict_sklearn(sequence)
        return None, 0.0
    
    def _predict_tf(self, sequence: np.ndarray) -> Tuple[str, float]:
        """TensorFlow LSTM prediction."""
        model = self.backend["clf_model"]
        input_data = np.expand_dims(sequence, axis=0)  # (1, T, F)
        probs = model.predict(input_data, verbose=0)[0]
        idx = np.argmax(probs)
        confidence = float(probs[idx]) * 100
        # Label mapping would come from the training metadata
        label = f"sign_{idx}"
        return label, confidence

    def _predict_sklearn(self, sequence: np.ndarray) -> Tuple[Optional[str], float]:
        """RandomForest prediction (v3 compat). Averages sequence to single vector."""
        if self.classifier is None:
            return None, 0.0
        
        features = np.mean(sequence, axis=0).reshape(1, -1)
        
        # Adapt feature count
        expected = self.classifier.n_features_in_
        actual = features.shape[1]
        if actual > expected:
            features = features[:, :expected]
        elif actual < expected:
            features = np.pad(features, ((0, 0), (0, expected - actual)))
        
        probs = self.classifier.predict_proba(features)[0]
        idx = np.argmax(probs)
        label = self.label_encoder.inverse_transform([idx])[0]
        return label, float(probs[idx]) * 100

    def predict_sentence(self, video_path: str, energy_threshold: float = 0.035) -> Tuple[Optional[List[str]], float]:
        """
        Temporal segmentation: splits video into signs by motion energy valleys.
        """
        full_seq = self.extract_from_video(video_path, max_frames=200)
        if full_seq is None or len(full_seq) < 10:
            return None, 0.0
        
        # Motion energy from hand region
        n_cols = min(full_seq.shape[1], 126)
        hand_motion = np.diff(full_seq[:, :n_cols], axis=0)
        energy = np.linalg.norm(hand_motion, axis=1)
        smooth = np.convolve(energy, np.ones(5) / 5, mode='same')
        is_moving = smooth > energy_threshold
        
        # Segment
        segments, in_seg, start = [], False, 0
        for i, mv in enumerate(is_moving):
            if mv and not in_seg:
                start = max(0, i - 5)
                in_seg = True
            elif not mv and in_seg:
                end = min(len(full_seq), i + 5)
                if end - start > 8:
                    segments.append(full_seq[start:end])
                in_seg = False
        if in_seg:
            segments.append(full_seq[start:])
        
        # Classify each segment
        results, total_conf = [], 0
        for seg in segments:
            label, conf = self.predict(seg)
            if label and conf > 45:
                results.append(label)
                total_conf += conf
        
        if not results:
            return None, 0.0
        return results, total_conf / len(results)

    # ═══════════════════════════════════════════
    # ─── VOCABULARY & TRAINING ───
    # ═══════════════════════════════════════════

    def _load_cached_landmarks(self):
        """Load all cached .npy landmark files from disk."""
        for npy in self.landmarks_dir.glob("*.npy"):
            self.landmark_dict[npy.stem] = np.load(npy)
    
    def _load_classifier(self):
        """Load the trained classifier from disk if available."""
        # TF model
        tf_path = self.models_dir / "sign_classifier.h5"
        if TF_AVAILABLE and tf_path.exists():
            try:
                self.backend["clf_model"] = tf.keras.models.load_model(str(tf_path))
                self.backend["classifier"] = "tensorflow"
                return
            except Exception:
                pass
        
        # Sklearn fallback
        pkl_path = self.models_dir / "psl_classifier.pkl"
        if pkl_path.exists() and self.classifier is None:
            try:
                with open(pkl_path, 'rb') as f:
                    self.classifier, self.label_encoder = pickle.load(f)
                self.backend["classifier"] = "sklearn"
            except Exception:
                pass

    def build_vocabulary(self, translator=None, word_map: Optional[Dict[str, str]] = None):
        """
        Build landmark dictionary from reference videos.
        Uses SLT translator if available, or processes local video files.
        """
        if word_map:
            for word, urdu in word_map.items():
                video = self.videos_dir / f"{word}.mp4"
                
                if not video.exists() and translator:
                    try:
                        clip = translator.translate(urdu)
                        clip.save(str(video), overwrite=True)
                    except Exception as e:
                        logger.warning(f"⚠️ Could not get video for '{word}': {e}")
                        continue
                
                if video.exists():
                    seq = self.extract_from_video(str(video))
                    if seq is not None:
                        self.landmark_dict[word] = seq
                        np.save(self.landmarks_dir / f"{word}.npy", seq)
                        logger.info(f"✅ Extracted DNA for: {word}")
                        # Sync to cloud
                        if self.gcs.client:
                            self.gcs.upload_file(str(self.landmarks_dir / f"{word}.npy"), f"landmarks/{word}.npy")

    def train(self, augment_count: int = 50) -> bool:
        """
        Train the classifier on the current vocabulary.
        Uses data augmentation for single-sample-per-class stability.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        if not self.landmark_dict:
            self._load_cached_landmarks()
        if not self.landmark_dict:
            return False
        
        X_aug, y_aug = [], []
        for word, seq in self.landmark_dict.items():
            mean_vec = np.mean(seq, axis=0)
            X_aug.append(mean_vec)
            y_aug.append(word)
            
            for _ in range(augment_count):
                noise = np.random.normal(0, 0.015, mean_vec.shape)
                scale = np.random.uniform(0.95, 1.05)
                X_aug.append((mean_vec + noise) * scale)
                y_aug.append(word)
        
        X = np.array(X_aug)
        y = np.array(y_aug)
        
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y)
        
        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=15,
            min_samples_split=5, random_state=42
        )
        self.classifier.fit(X, y_enc)
        self.backend["classifier"] = "sklearn"
        
        # Save
        with open(self.models_dir / "psl_classifier.pkl", 'wb') as f:
            pickle.dump((self.classifier, self.label_encoder), f)
        
        logger.info(f"✅ Trained on {len(self.landmark_dict)} words ({len(X)} samples)")
        
        # Sync to cloud
        if self.gcs.client:
            self.gcs.upload_file(str(self.models_dir / "psl_classifier.pkl"), "models/psl_classifier.pkl")
            
        return True

    def get_word_dna(self, word: str) -> Optional[np.ndarray]:
        """Get skeletal DNA for a word."""
        word = word.lower()
        if word in self.landmark_dict:
            return self.landmark_dict[word]
        path = self.landmarks_dir / f"{word}.npy"
        if path.exists():
            data = np.load(path)
            self.landmark_dict[word] = data
            return data
        return None

    def get_available_words(self) -> List[str]:
        """Return list of words with available landmark data."""
        on_disk = {p.stem for p in self.landmarks_dir.glob("*.npy")}
        return sorted(on_disk | set(self.landmark_dict.keys()))

    def export_dna_json(self, dna_sequence: np.ndarray) -> List[dict]:
        """Convert DNA to JSON for 3D rigging (Three.js/VRM)."""
        frames = []
        for frame_data in dna_sequence:
            kps = {}
            n_features = len(frame_data)
            
            if n_features >= 51:  # YOLO format
                for i in range(17):
                    kps[f"joint_{i}"] = {
                        "x": float(frame_data[i * 3]),
                        "y": float(frame_data[i * 3 + 1]),
                        "conf": float(frame_data[i * 3 + 2])
                    }
            else:  # MediaPipe format
                for i in range(min(n_features // 3, 33)):
                    kps[f"joint_{i}"] = {
                        "x": float(frame_data[i * 3]),
                        "y": float(frame_data[i * 3 + 1]),
                        "z": float(frame_data[i * 3 + 2])
                    }
            frames.append(kps)
        return frames
