import os
import json
import importlib

whisper = importlib.import_module("whisper")
import gradio as gr
import google.generativeai as genai
from gtts import gTTS

# ✅ Use the ffmpeg binary bundled with imageio-ffmpeg instead of requiring
# a system-wide install + PATH edit (run: pip install imageio-ffmpeg)
#
# NOTE: imageio_ffmpeg downloads a *versioned* exe name (e.g.
# "ffmpeg-win64-v7.0.2.exe"), but Whisper hardcodes the literal command
# "ffmpeg". Just adding the folder to PATH isn't enough — we also need a
# plain "ffmpeg.exe" copy sitting in that folder so Windows can find it.

# ---------- CONFIG ----------
ASSISTANT_NAME = "Iris"

# ❗ REPLACE WITH REAL KEY — must start with "AIza", get one at
# https://aistudio.google.com/apikey
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MEMORY_FILE = "iris_memory.json"
REPLY_AUDIO_PATH = "reply.mp3"

# gTTS language codes it supports; fall back to English if Whisper
# detects something gTTS doesn't have.
SUPPORTED_TTS_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "hi", "ja", "ko", "zh-CN",
    "ru", "ar", "nl", "tr", "pl", "sv", "id", "th", "vi",
}

# ---------- SPEECH TO TEXT ----------
print("Loading Whisper model (this happens once, may take a minute)...")
stt_model = whisper.load_model("small")

def transcribe(audio_path):
    if audio_path is None:
        return "", "en"
    result = stt_model.transcribe(audio_path, fp16=False)
    return result["text"].strip(), result.get("language", "en")

# ---------- MEMORY ----------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[memory] failed to load, starting fresh: {e}")
            return {"facts": [], "conversation": []}
    return {"facts": [], "conversation": []}

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f, indent=2)

memory = load_memory()

# ---------- LLM SETUP ----------
genai.configure(api_key=GOOGLE_API_KEY)

PERSONA = f"""You are {ASSISTANT_NAME}, a warm, quick-witted voice assistant.
Always reply in the same language the user just spoke or typed in.
Keep replies short and natural.
Never make up facts about the user.
"""

# ✅ Current, supported model name.
# gemini-pro and the entire gemini-1.5 line have been shut down.
# "gemini-flash-latest" always points to the newest Flash model if you'd
# rather not update this string every time Google retires one.
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=PERSONA,
)

chat = model.start_chat()

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

        if fact != "NONE" and fact not in memory["facts"]:
            memory["facts"].append(fact)
            save_memory(memory)

    except Exception as e:
        print(f"[fact extraction] error: {e}")

# ---------- GENERATE REPLY ----------
def generate_reply(user_text, lang_code="en"):
    try:
        extract_and_save_fact(user_text)

        response = chat.send_message(user_text)

        if not response or not hasattr(response, "text"):
            reply_text = "Sorry, I couldn't respond."
        else:
            reply_text = response.text

        memory["conversation"].append({
            "user": user_text,
            "assistant": reply_text
        })
        save_memory(memory)

        # ✅ Use the language that was actually detected, with a safe fallback
        tts_lang = lang_code if lang_code in SUPPORTED_TTS_LANGS else "en"
        try:
            tts = gTTS(text=reply_text, lang=tts_lang)
            tts.save(REPLY_AUDIO_PATH)
            audio_path = REPLY_AUDIO_PATH
        except Exception as e:
            print(f"[tts] error: {e}")
            audio_path = None

        return reply_text, audio_path

    except Exception as e:
        print(f"[generate_reply] error: {e}")
        return f"ERROR: {str(e)}", None

# ---------- AVATAR ----------
AVATAR_STATES = {
    "idle": "🤖",
    "listening": "👂",
    "thinking": "🤔",
    "speaking": "🗣️",
}

def avatar_html(state):
    return f"<div style='font-size:100px;text-align:center'>{AVATAR_STATES.get(state, '🤖')}</div>"

# ---------- RESPONSE HANDLERS ----------
def respond_from_audio(audio):
    if audio is None:
        yield "", "...", None, "\n".join(memory["facts"]), avatar_html("idle")
        return

    yield "...", "...", None, "\n".join(memory["facts"]), avatar_html("listening")

    user_text, detected_lang = transcribe(audio)

    yield user_text, "...", None, "\n".join(memory["facts"]), avatar_html("thinking")

    # ✅ Pass the detected language through instead of dropping it
    reply_text, audio_path = generate_reply(user_text, detected_lang)

    yield user_text, reply_text, audio_path, "\n".join(memory["facts"]), avatar_html("speaking")

def respond_from_text(user_text):
    if not user_text.strip():
        yield "", "...", None, "\n".join(memory["facts"]), avatar_html("idle")
        return

    yield user_text, "...", None, "\n".join(memory["facts"]), avatar_html("thinking")

    reply_text, audio_path = generate_reply(user_text)

    yield user_text, reply_text, audio_path, "\n".join(memory["facts"]), avatar_html("speaking")

# ---------- UI ----------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), title=ASSISTANT_NAME) as demo:
    gr.Markdown(f"# {ASSISTANT_NAME}")
    gr.Markdown("A voice assistant with memory — speak or type")

    with gr.Row():
        with gr.Column(scale=1):
            avatar = gr.HTML(avatar_html("idle"))
            memory_out = gr.Textbox(label="Memory", lines=6, interactive=False)

        with gr.Column(scale=2):
            with gr.Tab("🎤 Talk"):
                mic = gr.Audio(sources=["microphone"], type="filepath")
                mic_btn = gr.Button("Send")

            with gr.Tab("⌨️ Type"):
                text_in = gr.Textbox(label="Type here")
                text_btn = gr.Button("Send")

            user_out = gr.Textbox(label="You said")
            reply_out = gr.Textbox(label="Iris replied")
            audio_out = gr.Audio(autoplay=True)

    outputs = [user_out, reply_out, audio_out, memory_out, avatar]

    mic_btn.click(respond_from_audio, inputs=mic, outputs=outputs)
    text_btn.click(respond_from_text, inputs=text_in, outputs=outputs)
    text_in.submit(respond_from_text, inputs=text_in, outputs=outputs)

# ---------- RUN ----------
if __name__ == "__main__":
    demo.launch(debug=True)
