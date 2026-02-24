# Implementation Plan: Wasel v4 Pro — Cloud Run Optimization

This plan optimizes the Wasel v4 Pro engine and deployment configuration for 100% compatibility with Google Cloud Run, focusing on statelessness, performance, and scalability.

## Proposed Changes

### 1. Deployment Configuration [MODIFICATION]

#### [MODIFY] [Dockerfile](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/Dockerfile)
- Update `ENTRYPOINT` to use a shell script or dynamic command that respects the `$PORT` environment variable provided by Cloud Run.
- Optimize layer caching for faster builds.

#### [MODIFY] [requirements.txt](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/requirements.txt)
- Add `google-cloud-storage` for cloud persistence.
- Add `google-cloud-logging` for better observability.

### 2. Backend Orchestration [MODIFICATION]

#### [NEW] [gcp_utils.py](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/backend/gcp_utils.py)
- Create utility functions to sync the local `LANDMARKS_DIR` and `MODELS_DIR` with a GCS bucket.
- This ensures that training results from one instance are available to others (horizontal scaling).

#### [MODIFY] [engine.py](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/backend/engine.py)
- Integrate `gcp_utils` for automatic model/metadata sync.
- Enhance logging to use `google-cloud-logging` when running in a GCP environment.

#### [MODIFY] [app.py](file:///c:/Users/Ahmed/OneDrive%20-%20Konecta/Documents/mcp/wasel_v4_pro/app.py)
- Add a "Cloud Sync" indicator in the sidebar.
- Ensure temp files are cleaned up promptly to save memory.

## Verification Plan

### Automated Tests
- Build and run the Docker container locally using:
  `docker build -t wasel-v4-local .`
  `docker run -e PORT=8080 -p 8080:8080 wasel-v4-local`
- Verify that the app is accessible on the provided port.

### Manual Verification
- Deploy to Cloud Run using `gcloud run deploy`.
- Verify GCS bucket sync by training on one instance and seeing landmarks appear in the bucket.
