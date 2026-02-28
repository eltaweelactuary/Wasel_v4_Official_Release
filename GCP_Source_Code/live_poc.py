"""
Wasel v4 Pro: Live POC - Gradio Streaming Interface
Uses standard Gradio Image streaming (maximum compatibility, no WebRTC dependency).
"""
import os
import cv2
import numpy as np
import gradio as gr
from collections import deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Lazy engine loading: server starts FIRST, models load on first request
# This prevents Cloud Run timeout (container must listen on PORT within 300s)
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
    """Process a single frame from the webcam stream."""
    if image is None:
        return None

    eng = get_engine()
    if eng is None:
        cv2.putText(image, "Loading AI models...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return image

    try:
        if "pose_model" in eng.backend and eng.backend["pose_model"]:
            # Extract keypoints
            keypoints = eng.extract_keypoints_yolo(image)

            predicted_label = "Waiting for signs..."
            confidence = 0.0

            if keypoints is not None:
                sequence_buffer.append(keypoints)

                if len(sequence_buffer) > 5:
                    seq_array = np.array(list(sequence_buffer))
                    label, conf = eng.predict(seq_array)
                    if label and conf > 45.0:
                        predicted_label = f"{label} ({conf:.1f}%)"

            # Draw YOLO Pose annotations
            results = eng.backend["pose_model"](image, verbose=False)
            annotated_image = results[0].plot()

            # Draw prediction text
            cv2.putText(annotated_image, predicted_label, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Signing detection via motion energy
            if len(sequence_buffer) >= 2:
                seq = np.array(list(sequence_buffer))
                n_cols = min(seq.shape[1], 126)
                hand_motion = np.diff(seq[:, :n_cols], axis=0)
                energy = np.linalg.norm(hand_motion)
                if energy > 2.0:
                    cv2.putText(annotated_image, "Signing Detected", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            return annotated_image

    except Exception as e:
        logger.error(f"Error processing frame: {e}")

    return image

# Define Gradio Interface using standard Image streaming (no WebRTC dependency)
with gr.Blocks(title="Wasel v4 Live POC", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🤟 Wasel v4 Pro: Live Sign Language Translator
        Real-time Pakistan Sign Language translation powered by YOLOv8-Pose + LSTM.
        Allow camera access to begin.
        """
    )

    with gr.Row():
        input_img = gr.Image(sources=["webcam"], streaming=True, label="Live Camera Feed")
        output_img = gr.Image(label="AI Analysis Output")

    input_img.stream(
        fn=process_frame,
        inputs=[input_img],
        outputs=[output_img],
        time_limit=300,
        stream_every=0.1
    )

    gr.Markdown("Designed by Ahmed Eltaweel | AI Architect @ Konecta 🚀")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
