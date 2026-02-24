# Implementation Plan: Streamlit SLT App Optimization

## User Objective
Fix video streaming issues (latency/freezing) and enable true "online recognition" (real-time feedback) in the `streamlit_slt_app_v2_stable_v3_stable` (Legacy App).

## Current Issues
1.  **Video Stream Latency**: The `SignProcessor.recv` method performs heavy MediaPipe inference synchronously on the video thread. This causes frame drops and lag.
2.  **No True Online Recognition**: The current app only records frames and predicts *after* the user stops the stream. It does not provide real-time feedback during signing.
3.  **Audio/Mic Issues**: The "Speech to Text" feature relies on `speech_recognition` microphone access which fails on server environments without local audio hardware.

## Proposed Strategy (Antigravity Skills Applied)

### 1. Backend Engineer: Asynchronous Inference Pipeline
-   **Decouple Processing**: Move MediaPipe landmark extraction and Model Inference out of the critical `recv` loop.
-   **Threading**: Use a background thread to consume frames from a queue, run inference, and update a thread-safe "Latest Result" buffer.
-   **Optimization**: Downscale frames for inference (320x240) while keeping display high-res.

### 2. Frontend Engineer: Real-time Feedback Loop
-   **Overlay**: Draw the *prediction result* directly on the video frame in `recv` by reading the "Latest Result" buffer (which is updated by the background thread).
-   **Status Indicators**: Add visual cues (Green/Red borders) for "Hand Detected" vs "No Hand".

### 3. Solution Architect: Robust Error Handling
-   **Graceful Degrade**: If inference is too slow, skip frames rather than blocking the video stream.
-   **WebRTC Config**: Ensure ICE servers are correctly configured for maximum connectivity (already good, but verify).

## Implementation Steps

### Step 1: Refactor `SignProcessor` Class
-   [ ] Add `threading.Thread` for inference loop.
-   [ ] Implement `deque` for sliding window (temporal context).
-   [ ] Add `lock` for thread-safe access to prediction results.
-   [ ] Modify `recv` to only *push* frames to queue and *read* latest result for drawing.

### Step 2: Integrate 'Online' Logic
-   [ ] Instead of just "recording", run `core.predict_from_landmarks` continuously on the buffered sequence.
-   [ ] Display the predicted label on the video frame via `cv2.putText`.

### Step 3: Clean up UX
-   [ ] Remove 'Batch Analysis' logic from the Live Stream mode (since it will be real-time).
-   [ ] Add a "Reset" button to clear the sliding window.

## Verification
-   Run the app locally (`streamlit run legacy/app.py`).
-   Test "Live Stream" mode.
-   Verify video is smooth (30fps) and labels appear in real-time.
