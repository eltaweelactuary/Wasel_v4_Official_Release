"""
Wasel v4 Pro: WebRTC Streaming Hub
Real-time video streaming with AI inference overlay.
Replaces v3's broken streamlit-webrtc + MediaPipe combo.
"""

import cv2
import queue
import threading
import logging
import numpy as np
import av
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from streamlit_webrtc import VideoProcessorBase
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    # Stub for environments without streamlit-webrtc
    class VideoProcessorBase:
        def recv(self, frame):
            return frame


class SignStreamProcessor(VideoProcessorBase):
    """
    Non-blocking video processor for real-time sign language recognition.
    
    Architecture (fixes v3 frame-dropping issue):
        Main Thread: recv() → push frame to queue → draw overlay → return
        Background Thread: pop frame → YOLO/MP extraction → predict → update state
    
    This ensures video NEVER freezes regardless of inference speed.
    """

    def __init__(self, engine=None):
        self.engine = engine
        self.frame_queue = queue.Queue(maxsize=1)  # Drop old frames
        self.is_running = True

        # Thread-safe prediction state
        self.lock = threading.Lock()
        self.latest_prediction = "..."
        self.latest_confidence = 0.0
        self.frame_count = 0

        # Temporal context (sliding window for sentence-level recognition)
        self.history = deque(maxlen=30)  # ~1.5s @ 20fps

        # MediaPipe holistic instance (used only if engine backend is mediapipe)
        self._holistic = None
        if engine and engine.backend["pose"] == "mediapipe":
            try:
                import mediapipe as mp
                self._holistic = mp.solutions.holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                logger.warning(f"MP init failed: {e}")

        # Start background inference thread
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        """Background consumer: runs AI inference without blocking video."""
        while self.is_running:
            try:
                frame_rgb = self.frame_queue.get(timeout=0.5)

                if self.engine:
                    kp = self.engine.extract_keypoints(frame_rgb, self._holistic)
                    if kp is not None:
                        self.history.append(kp)

                        if len(self.history) >= 10:
                            window = np.array(list(self.history))
                            label, conf = self.engine.predict(window)

                            with self.lock:
                                if label and conf > 50:
                                    self.latest_prediction = label
                                    self.latest_confidence = conf
                                else:
                                    self.latest_prediction = "..."
                                    self.latest_confidence = 0.0
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Inference error: {e}")

    def recv(self, frame):
        """
        Main thread: receives video frame, pushes to queue, draws overlay.
        MUST return immediately to avoid video lag.
        """
        try:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # Push to background (non-blocking, drop if full)
            if not self.frame_queue.full():
                small = cv2.resize(img, (320, 240))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                self.frame_queue.put_nowait(rgb)

            # Draw HUD overlay
            with self.lock:
                pred = self.latest_prediction
                conf = self.latest_confidence

            h, w = img.shape[:2]
            color = (0, 255, 100) if pred != "..." else (80, 80, 80)

            # Top bar
            cv2.rectangle(img, (0, 0), (w, 55), (15, 15, 30), -1)
            cv2.putText(img, f"LIVE: {pred}", (15, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            if pred != "...":
                cv2.putText(img, f"{conf:.0f}%", (w - 80, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

            # Bottom status bar
            cv2.rectangle(img, (0, h - 30), (w, h), (15, 15, 30), -1)
            cv2.putText(img, f"Wasel v4 | Frame #{self.frame_count}", (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            return frame

    def __del__(self):
        self.is_running = False
        if self._holistic:
            self._holistic.close()
