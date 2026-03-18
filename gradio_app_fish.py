"""
Gradio interface for Fish Speech S2-Pro TTS.
Downloads model from https://huggingface.co/fishaudio/s2-pro on first run.
"""

import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from loguru import logger

# ── Model paths ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints/s2-pro")
HF_REPO = "fishaudio/s2-pro"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRECISION = torch.bfloat16

# ── Lazy globals ───────────────────────────────────────────────────────────────
_llama_model = None
_decode_one_token = None
_codec_model = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_models_downloaded():
    """Download model files from HuggingFace if not already present."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Check for required files
    required = ["model.pth", "codec.pth", "config.json", "tokenizer.tiktoken"]
    missing = [f for f in required if not (CHECKPOINT_DIR / f).exists()]

    if missing:
        logger.info(f"Downloading missing model files from {HF_REPO}: {missing}")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=str(CHECKPOINT_DIR),
                ignore_patterns=["*.md", "*.txt", "original/*"],
            )
            logger.info("Model download complete.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download model from {HF_REPO}. "
                f"You can also run:\n"
                f"  huggingface-cli download {HF_REPO} --local-dir {CHECKPOINT_DIR}\n"
                f"Error: {e}"
            )
    else:
        logger.info("Model files already present.")


def load_models():
    """Load LLaMA and codec models into memory (idempotent)."""
    global _llama_model, _decode_one_token, _codec_model

    if _llama_model is not None and _codec_model is not None:
        return

    ensure_models_downloaded()

    from fish_speech.models.text2semantic.inference import (
        init_model,
        load_codec_model,
    )

    if _llama_model is None:
        logger.info("Loading LLaMA model…")
        _llama_model, _decode_one_token = init_model(
            CHECKPOINT_DIR, DEVICE, PRECISION, compile=False
        )
        with torch.device(DEVICE):
            _llama_model.setup_caches(
                max_batch_size=1,
                max_seq_len=_llama_model.config.max_seq_len,
                dtype=next(_llama_model.parameters()).dtype,
            )
        logger.info("LLaMA model loaded.")

    if _codec_model is None:
        logger.info("Loading codec model…")
        _codec_model = load_codec_model(
            CHECKPOINT_DIR / "codec.pth", DEVICE, PRECISION
        )
        logger.info("Codec model loaded.")


# ── Inference ──────────────────────────────────────────────────────────────────

def generate_speech(
    text: str,
    reference_audio,
    reference_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    chunk_length: int,
    seed: int,
):
    if not text.strip():
        raise gr.Error("Please enter some text to synthesise.")

    load_models()

    from fish_speech.models.text2semantic.inference import (
        encode_audio,
        generate_long,
        decode_to_audio,
    )

    # ── Optional reference (voice cloning) ────────────────────────────────────
    prompt_tokens = None
    prompt_text = None

    if reference_audio is not None and reference_text.strip():
        logger.info("Encoding reference audio…")
        ref_path = Path(reference_audio)
        codes = encode_audio(ref_path, _codec_model, DEVICE)
        prompt_tokens = [codes.cpu()]
        prompt_text = [reference_text.strip()]
    elif reference_audio is not None:
        logger.warning("Reference audio provided but no reference text — ignoring reference.")
    elif reference_text.strip():
        logger.warning("Reference text provided but no reference audio — ignoring reference.")

    # ── Seed ──────────────────────────────────────────────────────────────────
    if seed > 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # ── Generate ──────────────────────────────────────────────────────────────
    logger.info(f"Generating speech for: {text[:80]}…")
    all_codes = []

    generator = generate_long(
        model=_llama_model,
        device=DEVICE,
        decode_one_token=_decode_one_token,
        text=text,
        num_samples=1,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        compile=False,
        iterative_prompt=chunk_length > 0,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    for response in generator:
        if response.action == "sample":
            all_codes.append(response.codes)
        elif response.action == "next":
            break

    if not all_codes:
        raise gr.Error("No audio was generated. Try adjusting the text or parameters.")

    merged_codes = torch.cat(all_codes, dim=1)
    audio_tensor = decode_to_audio(merged_codes.to(DEVICE), _codec_model)
    sample_rate = _codec_model.sample_rate

    audio_np = audio_tensor.float().cpu().numpy()
    logger.info(f"Generated {len(audio_np) / sample_rate:.2f}s of audio.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sample_rate, audio_np


# ── Gradio UI ──────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Fish Speech S2-Pro TTS") as demo:
        gr.Markdown(
            "# Fish Speech S2-Pro\n"
            "Text-to-speech using [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro). "
            "Models are downloaded automatically on first use."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesise…",
                    lines=4,
                )

                with gr.Accordion("Voice cloning (optional)", open=False):
                    gr.Markdown(
                        "Upload a reference audio clip and provide its transcript to clone a voice. "
                        "Both fields are required for voice cloning."
                    )
                    ref_audio = gr.Audio(
                        label="Reference audio",
                        type="filepath",
                    )
                    ref_text = gr.Textbox(
                        label="Reference transcript",
                        placeholder="Exact words spoken in the reference audio…",
                        lines=2,
                    )

                with gr.Accordion("Advanced parameters", open=False):
                    temperature = gr.Slider(
                        0.1, 1.0, value=0.8, step=0.05, label="Temperature"
                    )
                    top_p = gr.Slider(
                        0.1, 1.0, value=0.8, step=0.05, label="Top-p"
                    )
                    top_k = gr.Slider(
                        1, 100, value=30, step=1, label="Top-k"
                    )
                    max_new_tokens = gr.Slider(
                        0, 4096, value=1024, step=64,
                        label="Max new tokens (0 = unlimited)",
                    )
                    chunk_length = gr.Slider(
                        100, 300, value=200, step=50,
                        label="Chunk length (bytes)",
                    )
                    seed = gr.Number(
                        value=0, label="Seed (0 = random)", precision=0
                    )

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                audio_output = gr.Audio(label="Output audio", type="numpy")

        generate_btn.click(
            fn=generate_speech,
            inputs=[
                text_input,
                ref_audio,
                ref_text,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                chunk_length,
                seed,
            ],
            outputs=audio_output,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
