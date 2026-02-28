import os
import cv2
import numpy as np
import gradio as gr
from gradio_webrtc import WebRTC
from collections import deque
import logging

# Set up logging before importing engine
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Try to import the Wasel engine
try:
    from backend.engine import WaselEngine
    ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import WaselEngine: {e}")
    ENGINE_AVAILABLE = False

if ENGINE_AVAILABLE:
    logger.info("Initializing Wasel Engine...")
    engine = WaselEngine(
        data_dir="./wasel_v4_data",
        yolo_weights="yolov8n-pose.pt"
    )
    # Temporary buffer to hold sequential keypoints (e.g., last 30 frames)
    # The LSTM expects sequences of varying lengths, but we will maintain a rolling window.
    sequence_buffer = deque(maxlen=30)
else:
    engine = None
    sequence_buffer = deque()

def process_frame(image: np.ndarray) -> np.ndarray:
    """
    Process a single frame from the WebRTC stream.
    Image comes in as RGB numpy array.
    """
    if engine is None:
        # Fallback if engine fails to load: just return the image
        cv2.putText(image, "Engine Offline", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return image
        
    try:
        # YOLO works with RGB or BGR, we pass the raw image.
        # Ensure we have the model loaded
        if "pose_model" in engine.backend and engine.backend["pose_model"]:
            # Extract keypoints
            keypoints = engine.extract_keypoints_yolo(image)
            
            # Predict if we have keypoints
            predicted_label = "Waiting for signs..."
            confidence = 0.0
            
            if keypoints is not None:
                sequence_buffer.append(keypoints)
                
                # Only predict if we have accumulated enough frames (e.g. 5 or more)
                if len(sequence_buffer) > 5:
                    seq_array = np.array(list(sequence_buffer))
                    label, conf = engine.predict(seq_array)
                    if label and conf > 45.0:
                        predicted_label = f"{label} ({conf:.1f}%)"
            
            # Draw YOLO Pose annotations visually for the demo
            results = engine.backend["pose_model"](image, verbose=False)
            annotated_image = results[0].plot()
            
            # Draw prediction text on the frame
            cv2.putText(annotated_image, predicted_label, (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # Determine if the user is moving too much/too little (energy test)
            if len(sequence_buffer) >= 2:
                seq = np.array(list(sequence_buffer))
                hand_motion = np.diff(seq[:, :126], axis=0) # estimate based on engine logic
                energy = np.linalg.norm(hand_motion)
                if energy > 2.0: # arbitrary threshold for visual feedback
                    cv2.putText(annotated_image, "Signing Detected", (30, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                
            return annotated_image
            
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        
    return image

# Define Gradio Interface
with gr.Blocks(title="Wasel v4 Live POC") as demo:
    gr.Markdown(
        """
        # 🤟 Wasel v4 Pro: Live Translator (GCP WebRTC POC)
        Real-time Pakistan Sign Language translation using YOLO-Pose and Gradio WebRTC.
        Please allow camera access.
        """
    )
    
    with gr.Row():
        with gr.Column():
            # WebRTC Component optimized for continuous video stream
            video_stream = WebRTC(
                label="Live Camera Feed",
                mode="send-receive",
                modality="video",
            )
            
            # Wire the stream
            video_stream.stream(
                fn=process_frame,
                inputs=[video_stream],
                outputs=[video_stream],
                time_limit=180 # 3 minute demo sessions
            )
            
    gr.Markdown("Designed by Ahmed Eltaweel | AI Architect @ Konecta 🚀")

if __name__ == "__main__":
    # Get port from environment variables for Cloud Run compatibility
    port = int(os.environ.get("PORT", 8080))
    # Launch Gradio
    demo.launch(server_name="0.0.0.0", server_port=port)
