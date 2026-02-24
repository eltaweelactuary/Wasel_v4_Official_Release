# Walkthrough: Wasel v4 Pro — Cloud Run Optimization

This update transforms Wasel v4 Pro into a production-grade, cloud-native application ready for horizontal scaling on Google Cloud Run.

## 🚀 Key Improvements

### 1. Statelessness & Horizontal Scaling
Cloud Run instances are ephemeral. To ensure that signs trained on one instance are available to others, I implemented a custom GCS Synchronization layer.
- **File:** [gcp_utils.py](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/backend/gcp_utils.py)
- **Logic:** On startup, the engine downloads the latest landmarks and models from the configured GCS bucket. Any new training results are automatically uploaded.

### 2. Dynamic Port Handling
Cloud Run assigns a random port (usually 8080) to the container at runtime.
- **File:** [Dockerfile](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/Dockerfile)
- **Change:** Updated `ENTRYPOINT` to use `streamlit run ... --server.port=${PORT:-8501}`. This ensures the app always binds to the correct port.

### 3. Cloud-Native Observability
- **File:** [backend/engine.py](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/backend/engine.py)
- **Change:** Integrated `google-cloud-logging`. Logs are now structured and searchable in the GCP Logs Explorer.

### 4. Memory Optimization
Cloud Run memory is expensive and limited.
- **File:** [app.py](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/app.py)
- **Change:** Added aggressive temp file cleanup after video synthesis and upload processing to prevent memory leakage.

## 🛠️ Deployment Instructions

1.  **Set Environment Variables:**
    - `WASEL_GCS_BUCKET`: The name of your GCS bucket (e.g., `wasel-v4-assets`).
2.  **Deploy Command:**
    ```bash
    gcloud run deploy wasel-v4-pro \
      --source . \
      --region me-central1 \
      --memory 2Gi \
      --cpu 2 \
      --allow-unauthenticated
    ```

## ✅ Verification
- [x] Dockerfile port handling updated.
- [x] GCS Manager implemented and integrated.
- [x] Sidebar Cloud Status indicator added.
- [x] Memory cleanup logic verified.
