# Defensive FAQ: Wasel v4 Pro — Technical & Management Q&A

**Purpose:** Providing proactive answers to critical technical concerns during presentations.

---

## 🕒 Latency & Performance

### Q: You claim <100ms latency. How is this possible with Cloud-based AI?
**A:** We use a **Hybrid Execution Strategy**. While the system is *Cloud-Ready*, the core inference engine (YOLOv8 + TF) is hyper-optimized to run on the **Edge** (the user's browser/device) using WebAssembly or localized containers. The Cloud is used for heavy-duty training and data-sync (GCS), while the *Translation* happens as close to the camera as possible to bypass network lag.

### Q: What happens if the Camera FPS is faster than the AI Inference speed?
**A:** We implement an **Intelligent Frame Dropping** policy. The Frame Queue uses a "Last-In-First-Out" (LIFO) or "Skip-on-Busy" logic. If the AI thread is busy, it skips old frames and grabs the most recent one. This ensures we never develop a "lag drift" and the HUD always shows the *current* reality.

---

## 🔋 Hardware & Battery

### Q: Continuous AI detection will drain the battery quickly. How do we mitigate this?
**A:** We use **Activity Triggers**. The system runs in a "Low-Power Search Mode" (lower FPS) until a human shape is detected in front of the camera. Once active, it ramps up to "Full Inference Mode". After 30 seconds of inactivity, it returns to low-power state to preserve battery and prevent overheating.

---

## 🎯 Accuracy & Reliability

### Q: How do you prevent the system from translating random hand movements into text?
**A:** Our LSTM classifier includes a **"Null/Neutral Class"**. We train the model to recognize "noise" (like scratching a head or resting hands). Additionally, we use a **Confidence Threshold (e.g., 85%)**. If the AI isn't certain enough, it displays nothing on the HUD, preventing frustrating "False Positives".

### Q: What if the lighting is poor or the background is cluttered?
**A:** YOLOv8-Pose is significantly more robust than traditional MediaPipe in varying environments. Furthermore, our **Spatiotemporal DNA** extraction focuses only on joint relationships (vectors), making the system relatively immune to background colors or clutter.

---

## 🏗️ Scaling & Deployment

### Q: Why do we need Google Cloud Storage (GCS) if the processing can be local?
**A:** GCS acts as our **Shared Intelligence Hub**. It ensures that if we update a model or add a new word to the vocabulary, *every* device/instance updates automatically without requiring a full app reinstall. This is critical for maintaining a "Global Standard" in sign language translation.

### Q: Can this be deployed On-Premises for high-security sectors?
**A:** **Yes.** Because Wasel v4 Pro is built using Docker and a stateless API architecture, the entire stack is **technically deployable in minutes**. While institutional security compliance may take longer, our containerized architecture significantly accelerates the path to 100% data sovereignty.
placed by a local MinIO or Shared NAS, ensuring 100% data sovereignty.

---
> [!TIP]
> **Pitch Advice:** Always admit that Phase 1 is a "High-Efficiency Prototype". Mention that Phase 2 will include further hardware-specific optimizations (like TensorRT for NVIDIA or CoreML for Apple).
