# Documentation: Tech Stack & Project Roadmap (Phase 1)

**Project:** Wasel v4 Pro — Intelligent SLT  
**Stage:** Phase 1 (Core Engine & Live MVP)

---

## 🛠️ Technical Stack (Phase 1)

### 1. Core AI & Computer Vision
*   **Pose Estimation:** YOLOv8-Pose (Ultralytics) — Selected for superior FPS on CPU.
*   **Sequence Classification:** TensorFlow 2.15+ (LSTM / GRU) — Handles 30-frame temporal sign windows. Architecture designed for seamless upgrade to **Transformers** in Phase 2 for expanded vocabularies.
*   **Computer Vision Utils:** OpenCV-Python (Headless) — Frame preprocessing and HUD rendering.
*   **SLT Integration:** `sign-language-translator` — Used for Urdu/English text-to-gloss mapping.

### 2. Frontend & Streaming
*   **Web Framework:** Streamlit 1.30+ — Rapid UI development and state management.
*   **Real-time Video:** `streamlit-webrtc` — Low-latency P2P video streaming using STUN/TURN servers.
*   **Serialization:** Pickle/NumPy (.npy) — High-speed loading of skeletal DNA landmarks.

### 3. Cloud Infrastructure (GCP)
*   **Compute:** Google Cloud Run (Serverless) — Scales automatically based on request traffic.
*   **Persistence:** Google Cloud Storage (GCS) — Centralized repository for models and vocabulary DNA.
*   **Observability:** Google Cloud Logging — Structured JSON logging for engine diagnostics.

---

## 📅 Phase 1 Roadmap (4-Week Sprint)

### Week 1: Core Engine Stabilization
*   [ ] Refactoring `WaselEngine` to support dynamic backend switching.
*   [ ] Integration of GCS synchronization layer in `gcp_utils`.
*   [ ] Testing YOLO-to-TF pipeline with the first 5 core signs.

### Week 2: Real-time Streaming & HUD
*   [ ] Optimizing `webrtc_hub.py` for sub-50ms| **Inference Latency** | High: Video freezing. | **Non-blocking threading + LIFO Sampling Optimization:** Ensures UI stays at 30 FPS while AI thread captures only the most relevant motion keyframes. |
*   [ ] Designing the interactive HUD (Heads-Up Display) for live prediction feedback.
*   [ ] Implementing auto-setup scripts for model downloads on instance cold-starts.

### Week 3: Vocabulary Expansion (The "24 Golden Words")
*   [ ] Extracting landmark DNA for 24 core PSL words.
*   [ ] Training the temporal classifier with augmented datasets.
*   [ ] Validating recognition accuracy against the v3 benchmark.

### Week 4: Deployment & Documentation
*   [ ] CI/CD pipeline setup via Google Cloud Build.
*   [ ] Final stress testing on Cloud Run (Benchmarking memory/CPU limits).
*   [ ] Completion of User Guide and Technical Documentation.

---
> [!NOTE]
> This roadmap focuses on **Proof of Technical Feasibility**. Phase 2 will focus on UI/UX polishing and full vocabulary expansion.
