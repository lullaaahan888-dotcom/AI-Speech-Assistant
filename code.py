import io
import os
import tempfile

import streamlit as st
import whisper
import google.generativeai as genai
from gtts import gTTS

# ---------- CONFIG ----------
ASSISTANT_NAME = "Iris"

# gTTS language codes it supports; fall back to English if Whisper
# detects something gTTS doesn't have.
SUPPORTED_TTS_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "hi", "ja", "ko", "zh-CN",
    "ru", "ar", "nl", "tr", "pl", "sv", "id", "th", "vi",
}

PERSONA = f"""You are {ASSISTANT_NAME}, a warm, quick-witted voice assistant.
Always reply in the same language the user just spoke or typed in.
Keep replies short and natural.
Never make up facts about the user.
"""

st.set_page_config(page_title=ASSISTANT_NAME, page_icon="🤖")

# ---------- CACHED RESOURCES (loaded once per server, not per user) ----------
@st.cache_resource(show_spinner="Loading speech-to-text model...")
def load_whisper():
    # "base" keeps memory use reasonable on Streamlit Community Cloud's
    # free tier. Bump to "small" or "medium" if you're on a paid plan
    # with more RAM.
    return whisper.load_model("base")


@st.cache_resource(show_spinner="Connecting to Gemini...")
def load_model():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error(
            "No GOOGLE_API_KEY found in Streamlit secrets. "
            "Add it under App settings → Secrets."
        )
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=PERSONA,
    )


stt_model = load_whisper()
model = load_model()

# ---------- SESSION STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, audio?}]
if "facts" not in st.session_state:
    st.session_state.facts = []
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat()
if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None


# ---------- SPEECH TO TEXT ----------
def transcribe(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        result = stt_model.transcribe(path, fp16=False)
        return result["text"].strip(), result.get("language", "en")
    finally:
        os.remove(path)


# ---------- SAFE FACT EXTRACTION ----------
def extract_and_save_fact(user_text):
    try:
        check_prompt = (
            "Does this message reveal a durable personal fact worth remembering? "
            "Reply with ONE short sentence or NONE.\n\n"
            f"Message: {user_text}"
        )
        response = model.generate_content(check_prompt)
        if not response or not hasattr(response, "text"):
            return
        fact = response.text.strip()
        if fact != "NONE" and fact not in st.session_state.facts:
            st.session_state.facts.append(fact)
    except Exception as e:
        print(f"[fact extraction] error: {e}")


# ---------- GENERATE REPLY ----------
def generate_reply(user_text, lang_code="en"):
    extract_and_save_fact(user_text)

    try:
        response = st.session_state.chat.send_message(user_text)
        reply_text = response.text if hasattr(response, "text") else "Sorry, I couldn't respond."
    except Exception as e:
        print(f"[generate_reply] error: {e}")
        return f"ERROR: {e}", None

    tts_lang = lang_code if lang_code in SUPPORTED_TTS_LANGS else "en"
    audio_bytes = None
    try:
        tts = gTTS(text=reply_text, lang=tts_lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        audio_bytes = buf.getvalue()
    except Exception as e:
        print(f"[tts] error: {e}")

    return reply_text, audio_bytes


# ---------- SIDEBAR ----------
with st.sidebar:
    st.subheader("🧠 Memory")
    if st.session_state.facts:
        for fact in st.session_state.facts:
            st.write(f"- {fact}")
    else:
        st.caption("No facts saved yet.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat = model.start_chat()
        st.rerun()

# ---------- MAIN ----------
st.title(f"{ASSISTANT_NAME} 🤖")
st.caption("A voice assistant with memory — speak or type")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

audio_value = st.audio_input("🎤 Record a message")
text_value = st.chat_input("Or type here")

user_text, detected_lang = None, "en"

if audio_value is not None:
    audio_id = hash(audio_value.getvalue())
    if audio_id != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio_id
        with st.spinner("Transcribing..."):
            user_text, detected_lang = transcribe(audio_value.getvalue())

if text_value:
    user_text, detected_lang = text_value, "en"

if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)

    with st.spinner("Thinking..."):
        reply_text, audio_bytes = generate_reply(user_text, detected_lang)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply_text, "audio": audio_bytes}
    )
    with st.chat_message("assistant"):
        st.write(reply_text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)
