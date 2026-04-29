#!/usr/bin/env python3
"""
LTX-Video direkte Inferenz-Pipeline für Intel Mac mit PyTorch 2.2.2
Umgeht ComfyUI und nutzt diffusers + LTX-Video safetensors direkt.

Strategie:
- T5 Text-Encoder: t5xxl_fp8_e4m3fn.safetensors → manuell zu bfloat16 dequantisiert
- T5 Config: aus lokalem HF-Cache (diffusers/LTX-Video-0.9.1)
- Transformer + VAE: ltxv-2b-0.9.8-distilled-fp8.safetensors via from_single_file
"""

import sys
import os
import json
import time
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Kompatibilitäts-Patches für PyTorch 2.2.2 ──────────────────────
# torch.xpu existiert erst ab 2.3
if not hasattr(torch, 'xpu'):
    class _XPUStub:
        def empty_cache(self): pass
        def synchronize(self): pass
        def is_available(self): return False
        def device_count(self): return 0
        def current_device(self): return 0
        def manual_seed(self, seed): pass
        def get_device_name(self, idx=0): return ""
        def mem_get_info(self): return (0, 0)
    torch.xpu = _XPUStub()

# ── Imports ──────────────────────────────────────────────────────────
from safetensors import safe_open
from transformers import T5EncoderModel, T5Config, T5Tokenizer
from diffusers import LTXPipeline
from diffusers.utils import export_to_video


MODEL_PATH = "/Users/samsimac/ComfyUI/models/checkpoints/ltxv-2b-0.9.8-distilled-fp8.safetensors"
T5_PATH = "/Users/samsimac/ComfyUI/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors"
T5_CONFIG_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--diffusers--LTX-Video-0.9.1"
    "/snapshots/a248b564f0d86b8537474efe2b9f0e05a63e087c/text_encoder"
)
TOKENIZER_PATH = "/Users/samsimac/ComfyUI/models/text_encoders/tokenizer"
OUTPUT_DIR = "/Users/samsimac/ComfyUI/output"


def _load_t5_encoder() -> T5EncoderModel:
    """Lade T5-xxl aus fp8-Safetensors, dequantisiere zu bfloat16."""
    logger.info("Lade T5-Config...")
    config = T5Config.from_pretrained(T5_CONFIG_PATH)

    logger.info("Erstelle leeres T5-Modell (bfloat16)...")
    model = T5EncoderModel(config).to(torch.bfloat16)

    logger.info("Lade fp8-Gewichte und dequantisiere zu bfloat16...")
    state = {}
    with safe_open(T5_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype == torch.float8_e4m3fn:
                t = t.to(torch.bfloat16)
            state[k] = t

    model.load_state_dict(state, strict=True)
    logger.info(f"T5 geladen ({sum(p.numel() for p in model.parameters())/1e9:.1f}B Parameter)")
    return model


def load_pipeline():
    """Lade vollständige LTX-Video Pipeline."""
    text_encoder = _load_t5_encoder()

    logger.info("Lade T5-Tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)

    logger.info("Lade LTX-Video 0.9.8 fp8-Checkpoint (Transformer + VAE)...")
    pipe = LTXPipeline.from_single_file(
        MODEL_PATH,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cpu")

    logger.info("Pipeline bereit.")
    return pipe


def generate_video(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    num_frames: int = 25,
    width: int = 256,
    height: int = 192,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = 42,
    output_path: str = None,
) -> str:
    """Generiere ein Video und speichere es."""

    pipe = load_pipeline()

    generator = torch.Generator("cpu").manual_seed(seed)

    logger.info(f"Generiere Video: '{prompt[:80]}'...")
    logger.info(f"Frames: {num_frames}, Größe: {width}x{height}, Steps: {num_inference_steps}")

    start_time = time.time()

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )

    elapsed = time.time() - start_time
    logger.info(f"Generierung abgeschlossen in {elapsed:.1f}s")

    # Output speichern
    if output_path is None:
        timestamp = int(time.time())
        output_path = os.path.join(OUTPUT_DIR, f"ltx_{timestamp}.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    export_to_video(output.frames[0], output_path, fps=8)

    logger.info(f"Video gespeichert: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LTX-Video Inferenz")
    parser.add_argument("--prompt", required=True, help="Text-Prompt")
    parser.add_argument("--negative_prompt", default="worst quality, blurry, distorted")
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    result = generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.frames,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        seed=args.seed,
        output_path=args.output,
    )

    print(json.dumps({"status": "success", "output_path": result}))
