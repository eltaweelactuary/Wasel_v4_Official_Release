"""
Wasel v4 Pro: Live POC - FastRTC WebRTC Streaming
Uses fastrtc Stream API for immediate, no-click live webcam streaming.
"""
import os
import cv2
import numpy as np
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Lazy engine loading: server starts FIRST, models load on first request
engine = None
sequence_buffer = deque(maxlen=30)
_engine_loaded = False

def get_engine():
    global engine, _engine_loaded
    if _engine_loaded:
        return engine
    _engine_loaded = True
    try:
        from backend.engine import WaselEngine
        logger.info("Initializing Wasel Engine (first request)...")
        engine = WaselEngine(
            data_dir="./wasel_v4_data",
            yolo_weights="yolov8n-pose.pt"
        )
        logger.info("Engine ready!")
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        engine = None
    return engine

def process_frame(image):
    """Process a single frame from the WebRTC stream. Returns annotated frame."""
    if image is None:
        return None

    eng = get_engine()
    if eng is None:
        # Show loading message while models are loading
        overlay = image.copy()
        cv2.putText(overlay, "Loading AI models...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return overlay

    try:
        if "pose_model" in eng.backend and eng.backend["pose_model"]:
            # Run YOLO Pose detection and draw skeleton
            results = eng.backend["pose_model"](image, verbose=False)
            annotated_image = results[0].plot()

            # Extract keypoints for prediction
            keypoints = eng.extract_keypoints_yolo(image)
            predicted_label = "Waiting for signs..."

            if keypoints is not None:
                sequence_buffer.append(keypoints)

                if len(sequence_buffer) > 5:
                    seq_array = np.array(list(sequence_buffer))
                    label, conf = eng.predict(seq_array)
                    if label and conf > 45.0:
                        predicted_label = f"{label} ({conf:.1f}%)"

            # Draw prediction text
            cv2.putText(annotated_image, predicted_label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Motion energy detection
            if len(sequence_buffer) >= 2:
                seq = np.array(list(sequence_buffer))
                n_cols = min(seq.shape[1], 126)
                hand_motion = np.diff(seq[:, :n_cols], axis=0)
                energy = np.linalg.norm(hand_motion)
                if energy > 2.0:
                    cv2.putText(annotated_image, "Signing Detected", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            return annotated_image
        else:
            # No pose model, just show the raw frame with a message
            cv2.putText(image, "YOLO model not loaded", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image

    except Exception as e:
        logger.error(f"Error processing frame: {e}")

    return image

if __name__ == "__main__":
    from fastrtc import Stream

    port = int(os.environ.get("PORT", 8080))

    stream = Stream(
        handler=process_frame,
        modality="video",
        mode="send-receive",
    )

    logger.info(f"Starting Wasel v4 Live POC on port {port}...")
    stream.ui.launch(server_name="0.0.0.0", server_port=port)
