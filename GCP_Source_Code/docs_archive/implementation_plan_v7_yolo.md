# Wasel v4 Pro: Complete Rebuild Plan

## 1. ما هي مشاكل v3 الجذرية؟

### 🔴 مشاكل حرجة (تمنع الإنتاج)

| # | المشكلة | السبب الجذري |
|---|---|---|
| 1 | **تقطيع الفيديو المباشر** | `streamlit-webrtc` بدون TURN server حقيقي، MediaPipe ثقيل في الخيط الرئيسي |
| 2 | **خطأ 229 vs 225 features** | `extract_frame_features` ينتج 229 لكن الـ RandomForest المحفوظ يتوقع 225 |
| 3 | **mediapipe version crash** | الإصدار الأخير أزال `solutions.holistic` — تم تثبيته على 0.10.14 كحل مؤقت |
| 4 | **ملف `app.py` = 1270 سطر** | كل شيء في ملف واحد: UI + Logic + Patches + CSS + Config |
| 5 | **Monkeypatching الملفات** | 100+ سطر لتحويل مسارات الملفات لأن `slt` library تكتب في أماكن محمية |
| 6 | **9 كلمات فقط** | القاموس ثابت يدوياً ولا يتوسع تلقائياً |

### 🟡 مشاكل وظيفية

| # | المشكلة |
|---|---|
| 7 | البشري الرقمي (Three.js) لا يتزامن مع الصوت |
| 8 | لا يوجد نظام تدريب تلقائي (AutoML) |
| 9 | لا يوجد Dockerfile يعمل على Cloud Run |

---

## 2. المقارنة مع المعيار العالمي

### الأبحاث المنشورة

| المشروع | التقنية | الدقة | زمن الاستجابة | المصدر |
|---|---|---|---|---|
| **Google PopSign** | TF Lite + MediaPipe | 97%+ | < 20ms | [tensorflow.org](https://www.tensorflow.org/blog/asl-fingerspelling) |
| **Google SignTown** | TF.js + PoseNet | 95%+ | Real-time | [blog.google](https://blog.google/outreach-initiatives/accessibility/signtownproject/) |
| **YOLOv8 + MediaPipe (IEEE)** | YOLOv8s + MP | **98% mAP** | **18ms/frame** | [IEEE Xplore](https://ieeexplore.ieee.org) |
| **YOLOv8 ASL (ResearchGate)** | YOLOv8n-pose | **97.21%** | 25 FPS | [AmericasPG](https://americaspg.com) |
| **TF Object Detection ISL** | SSD MobileNet v2 | 87.4% | ~50ms | [IRJET](https://www.irjet.net) |

### مقارنة Wasel v3 مقابل v4 المقترح

| البعد | v3 (الحالي) | v4 Pro (المقترح) | المعيار العالمي |
|---|---|---|---|
| **المحرك** | MediaPipe + RandomForest | **YOLOv8-Pose + TF LSTM** | TF / YOLO / MP |
| **الدقة** | ~60-70% (9 كلمات) | **95%+ (قابلة للتوسع)** | 95-98% |
| **السرعة** | ~200ms + تقطيع | **< 30ms / frame** | < 20ms |
| **القاموس** | 9 كلمات ثابتة | **Roboflow Pipeline (مفتوح)** | آلاف الكلمات |
| **البنية** | ملف واحد 1270 سطر | **Modular (6 ملفات)** | Microservices |
| **النشر** | Streamlit Cloud فقط | **GCP Vertex AI + Cloud Run** | Cloud + Edge |
| **البشري الرقمي** | Three.js غير متزامن | **WebRTC Sync + VRM** | WebRTC + 3D |

---

## 3. هيكل المشروع الجديد

```
wasel_v4_pro/
├── app.py                    # واجهة Streamlit (< 200 سطر)
├── requirements.txt
├── packages.txt
├── Dockerfile
├── .env.example
│
├── backend/
│   ├── __init__.py
│   ├── engine.py             # المحرك الرئيسي: YOLO + TF
│   ├── vocabulary.py         # إدارة القاموس الديناميكي
│   └── digital_human.py      # محرك البشري الرقمي (منقول + محسن من v3)
│
├── streaming/
│   ├── __init__.py
│   └── webrtc_hub.py         # WebRTC + Roboflow Inference SDK
│
├── models/
│   ├── yolov8n-pose.pt       # أوزان YOLO للهيكل العظمي
│   └── sign_classifier.h5    # مصنّف TensorFlow المدرّب
│
├── deployment/
│   ├── vertex_ai_config.yaml
│   └── cloudbuild.yaml
│
├── static/                   # منقول من v3
│   ├── app.js
│   ├── three_bridge.js
│   └── style.css
│
└── assets/                   # منقول من v3
    └── *.vrm
```

---

## 4. خطة التنفيذ (مرتبة حسب الأولوية)

### المرحلة 1: المحرك الأساسي ⚙️
1. `backend/engine.py` — YOLOv8-Pose + TensorFlow LSTM classifier
2. `backend/vocabulary.py` — قاموس ديناميكي يُحمّل من Roboflow أو ملفات محلية
3. `backend/__init__.py`

### المرحلة 2: البث والتدفق 📡
4. `streaming/webrtc_hub.py` — WebRTC مع Roboflow Inference SDK
5. `app.py` — واجهة Streamlit نظيفة وقصيرة

### المرحلة 3: البشري الرقمي 🧍
6. `backend/digital_human.py` — نقل `DigitalHumanRenderer` من v3 + تزامن WebRTC

### المرحلة 4: النشر ☁️
7. `Dockerfile` + `deployment/` — GCP Cloud Run + Vertex AI
8. `.env.example` — إدارة المفاتيح

---

## 5. خطة التحقق

- **Automated:** اختبار زمن الاستجابة (< 50ms) ودقة التصنيف (> 90%)
- **Manual:** تجربة البث المباشر في ظروف إضاءة مختلفة
- **Comparison:** مقارنة مباشرة بين v3 و v4 على نفس الفيديو
