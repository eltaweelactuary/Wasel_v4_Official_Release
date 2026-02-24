"""
Wasel v4 Pro: Intelligent Sign Language Translator
Production-ready Streamlit app for Google Cloud Run.

Workflow:
  1. Text → PSL Video (Digital Human Avatar)
  2. Video Upload → Text (Temporal Segmentation + Classification)
  3. Live Camera → Real-time Recognition (WebRTC + Background Inference)
"""

import os
import tempfile
import logging
import streamlit as st

# ─── Page Config (must be first Streamlit call) ───
st.set_page_config(page_title="Wasel v4 Pro", page_icon="🤟", layout="wide")

# ─── Logging ───
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Environment ───
DATA_DIR = os.environ.get("WASEL_DATA_DIR", os.path.join(tempfile.gettempdir(), "wasel_v4_data"))
os.makedirs(DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════════
# ─── CACHED RESOURCE LOADERS ───
# ═══════════════════════════════════════════

@st.cache_resource
def get_engine():
    from backend.engine import WaselEngine
    legacy_pkl = os.path.join(os.getcwd(), "assets", "psl_classifier.pkl")
    engine = WaselEngine(
        data_dir=DATA_DIR,
        legacy_model_path=legacy_pkl if os.path.exists(legacy_pkl) else None
    )
    return engine


@st.cache_resource
def get_vocabulary():
    from backend.vocabulary import VocabularyManager
    return VocabularyManager(include_extended=True)


@st.cache_resource
def get_renderer():
    from backend.digital_human import DigitalHumanRenderer
    return DigitalHumanRenderer()


def auto_setup(engine, vocab):
    """Auto-build vocabulary and train classifier on first run."""
    if engine.classifier and len(engine.landmark_dict) > 0:
        return True

    with st.status("🚀 **First-Time Setup** — Building AI Engine...", expanded=True) as status:
        if len(engine.landmark_dict) == 0:
            st.write("📥 **Step 1/3:** Downloading PSL reference videos...")
            try:
                import sign_language_translator as slt
                translator = slt.models.ConcatenativeSynthesis(
                    text_language="urdu", sign_language="psl", sign_format="vid"
                )
                engine.build_vocabulary(translator=translator, word_map=vocab.get_core())
            except Exception as e:
                st.write(f"⚠️ SLT Library unavailable ({e}). Using cached data if available.")
            st.write(f"✅ DNA extracted for **{len(engine.landmark_dict)}** words.")
        else:
            st.write(f"✅ **Step 1/3:** {len(engine.landmark_dict)} words loaded from cache.")

        if not engine.classifier:
            st.write("🧠 **Step 2/3:** Training classifier...")
            engine.train()
            st.write("✅ Classifier trained and saved.")
        else:
            st.write("✅ **Step 2/3:** Classifier loaded from cache.")

        words = engine.get_available_words()
        st.write(f"🔍 **Step 3/3:** {len(words)} words ready: {', '.join(words)}")
        status.update(label="✅ **Setup Complete!**", state="complete")
    return True


# ═══════════════════════════════════════════
# ─── MAIN APP ───
# ═══════════════════════════════════════════

def main():
    # ─── Custom CSS ───
    st.markdown("""<style>
        .main { background-color: #0f172a; }
        h1, h2, h3 { color: #38bdf8 !important; }
        .stButton>button { border-radius: 10px; border: 1px solid #4f46e5; }
    </style>""", unsafe_allow_html=True)

    st.title("🤟 Wasel v4 Pro")
    st.markdown("**Next-Gen Pakistan Sign Language Translator** — YOLO + TensorFlow")
    st.divider()

    # ─── Load Engines ───
    with st.spinner("⏳ Loading AI engines..."):
        engine = get_engine()
        vocab = get_vocabulary()
        renderer = get_renderer()

    if not auto_setup(engine, vocab):
        st.error("❌ Setup failed.")
        st.stop()

    # ─── Sidebar ───
    st.sidebar.success(f"💎 **Wasel v4 Pro** | Elite Studio")
    st.sidebar.success(f"🧠 **Engine:** {engine.backend['pose']} + {engine.backend['classifier']}")

    if engine.gcs.client:
        st.sidebar.info(f"☁️ **Cloud Storage:** Connected")
    else:
        st.sidebar.warning("💾 **Mode:** Local Storage")

    st.sidebar.info(f"📚 **Vocabulary:** {len(engine.get_available_words())} words")

    # ─── Tabs ───
    tab_text, tab_video, tab_live = st.tabs(["📝 Text → Video", "🎥 Video → Text", "🌐 Live Stream"])

    # ═══════════════════════════════════════
    # TAB 1: TEXT → PSL VIDEO (AVATAR)
    # ═══════════════════════════════════════
    with tab_text:
        st.subheader("✍️ Translate English to PSL Avatar")
        text_input = st.text_input("Enter English text:", placeholder="good apple pakistan")

        if st.button("🔄 Translate to PSL", key="translate_btn"):
            if text_input:
                words = text_input.lower().split()
                dna_list = []
                found_words = []

                for w in words:
                    dna = engine.get_word_dna(w)
                    if dna is not None:
                        dna_list.append(dna)
                        found_words.append(w)
                        st.success(f"✅ {w}")
                    else:
                        st.warning(f"⚠️ '{w}' not in vocabulary. Skipping.")

                if dna_list:
                    out_path = os.path.join(tempfile.gettempdir(), "wasel_v4_output.mp4")
                    renderer.stitch_and_render(dna_list, out_path)

                    # Read file into memory BEFORE displaying
                    with open(out_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                    st.caption(f"🧍 Avatar performing: **{' → '.join(found_words)}**")

                    # Now safe to cleanup
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass

    # ═══════════════════════════════════════
    # TAB 2: VIDEO → TEXT (UPLOAD)
    # ═══════════════════════════════════════
    with tab_video:
        st.subheader("📁 Upload a Sign Language Video")
        uploaded = st.file_uploader("Upload sign clip:", type=["mp4", "avi", "mov"])

        if uploaded:
            tmp = os.path.join(tempfile.gettempdir(), "wasel_v4_input.mp4")
            with open(tmp, "wb") as f:
                f.write(uploaded.read())

            # Show the uploaded video
            st.video(uploaded)

            with st.spinner("🧠 Analyzing signs..."):
                labels, conf = engine.predict_sentence(tmp)

            # Cleanup temp file
            try:
                os.remove(tmp)
            except OSError:
                pass

            if labels:
                st.success(f"🎯 **Recognized:** {' '.join(labels)}")
                st.metric("Confidence", f"{conf:.1f}%")

                # Show reverse: recognized text → Avatar
                st.divider()
                st.subheader("🧍 Avatar Replay")
                dna_list = [engine.get_word_dna(w) for w in labels if engine.get_word_dna(w) is not None]
                if dna_list:
                    replay_path = os.path.join(tempfile.gettempdir(), "wasel_v4_replay.mp4")
                    renderer.stitch_and_render(dna_list, replay_path)
                    with open(replay_path, "rb") as vf:
                        st.video(vf.read())
                    try:
                        os.remove(replay_path)
                    except OSError:
                        pass
            else:
                st.warning("⚠️ No signs detected. Try a clearer video.")

    # ═══════════════════════════════════════
    # TAB 3: LIVE CAMERA STREAM
    # ═══════════════════════════════════════
    with tab_live:
        st.subheader("🌐 Real-time Sign Language Recognition")
        st.markdown("Point your camera at a signer. Recognition results appear as an overlay.")

        try:
            from streamlit_webrtc import webrtc_streamer
            from streaming.webrtc_hub import SignStreamProcessor

            webrtc_streamer(
                key="wasel_v4_live",
                video_processor_factory=lambda: SignStreamProcessor(engine=engine),
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )

            st.info("💡 Signs are recognized in real-time. The overlay shows the predicted sign and confidence.")
        except ImportError:
            st.info("📦 `streamlit-webrtc` is required for live streaming. Install it with: `pip install streamlit-webrtc`")
        except Exception as e:
            st.error(f"❌ Stream error: {e}")

    # ─── Footer ───
    st.divider()
    st.caption("Designed by Ahmed Eltaweel | AI Architect @ Konecta 🚀")


if __name__ == "__main__":
    main()
