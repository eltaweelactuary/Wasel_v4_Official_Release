# Documentation: Risk Assessment & QA Strategy

**Project:** Wasel v4 Pro  
**Stage:** Phase 1 (First Stage Submission)

---

## âš ï¸ Risk Assessment & Mitigation

| Risk Category | Potential Impact | Mitigation Strategy |
|---|---|---|
| **Inference Latency** | High: Video freezing. | **Non-blocking threading + LIFO Sampling Optimization:** Ensures UI stays at 30 FPS while AI thread captures only the most relevant motion keyframes. |
| **Model Drift** | Medium: Accuracy drops in different lighting. | **Data Augmentation:** Injecting noise, scaling, and rotation variations during the training of the 24-word classifier. |
| **Cloud Cold Start** | Low: Initial load time for the user. | **GCS Sync:** Pre-fetching only essential model weights on startup and using Streamlit's `@cache_resource`. |
| **Service Permissions** | Medium: Deployment failures on GCP. | **Robust gcloud setup:** Using specific service accounts (`sa-vertex`) with least-privilege roles. |

---

## ðŸ§ª Quality Assurance (QA) Strategy

### 1. Recognition Accuracy Benchmarking
*   **KPI:** Minimum **95% accuracy** on the 24-word core vocabulary.
*   **Method:** Cross-validation on a hold-out set of skeletal DNA extracted from different video sources.

### 2. Performance Testing
*   **Target:** Inference time **< 100ms** per window.
*   **Metric:** Frame processing time logged in GCP Logs Explorer.

### 3. User Experience (UX) Validation
*   **Manual Test:** Verifying that the Digital Human Avatar renders transitions between words in less than **500ms**.
*   **Connectivity:** Ensuring WebRTC P2P connection establishes within **3 seconds** of starting the camera.

### 4. Regression Testing
*   **Fallback Check:** Manually disabling YOLO to verify the engine automatically switches to MediaPipe without crashing the UI.

---
> [!IMPORTANT]
> All QA results will be documented in a **Phase 1 Validation Report** before moving to Phase 2.

