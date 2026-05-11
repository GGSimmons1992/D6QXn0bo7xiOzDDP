import os
import numpy as np
import keras
import gradio as gr
from dotenv import load_dotenv, find_dotenv

try:
    import librosa
except ImportError as e:
    raise ImportError(
        "librosa is required. Install with `pip install librosa soundfile`"
    ) from e

_ = load_dotenv(find_dotenv())  # read local .env file

# ── MFCC inference settings (must match training pipeline) ───────────────────
SR = 22050
N_MFCC = 40
MAX_LEN = 200

# ── Resolve model path relative to this file ─────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_SRC_DIR, "..", "Models", "best_dv_model_.h5")

model = keras.models.load_model(_MODEL_PATH, compile=False)
model.summary()


def preprocess_audio(audio_path: str) -> np.ndarray:
    """
    Load an audio file and return an MFCC tensor shaped (1, MAX_LEN, N_MFCC, 1),
    matching the preprocessing applied during training.
    """
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)  # (N_MFCC, frames)
    mfcc = mfcc.T  # (frames, N_MFCC)

    frames = mfcc.shape[0]
    if frames >= MAX_LEN:
        mfcc_fixed = mfcc[:MAX_LEN, :]
    else:
        pad = np.zeros((MAX_LEN - frames, N_MFCC), dtype=mfcc.dtype)
        mfcc_fixed = np.vstack([mfcc, pad])

    mfcc_fixed = mfcc_fixed.astype(np.float32)
    # (1, MAX_LEN, N_MFCC, 1)
    return np.expand_dims(np.expand_dims(mfcc_fixed, axis=0), axis=-1)


def detectFakeAudio(audio_path: str) -> str:
    """Gradio prediction function: returns 'fake audio' or 'real audio'."""
    if audio_path is None:
        return "No audio provided."
    try:
        tensor = preprocess_audio(audio_path)
    except Exception as e:
        return f"Error processing audio: {e}"

    try:
        score = float(model.predict(tensor)[0, 0])
    except Exception as e:
        return f"Error running model: {e}"

    print(f"Fake-audio score: {score:.4f}")
    return "fake audio" if score > 0.5 else "real audio"


def main():
    gr.close_all()
    demo = gr.Interface(
        fn=detectFakeAudio,
        inputs=[gr.Audio(label="Upload audio file", type="filepath")],
        outputs=[gr.Textbox(label="Detection result")],
        title="Fake Audio Detector",
        description="Upload an audio clip to detect whether it is AI-generated (fake) or genuine (real).",
        flagging_mode="never",
    )
    demo.launch(share=True, server_port=int(os.getenv("PORT1", "7860")))


if __name__ == "__main__":
    main()
