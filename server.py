#!/usr/bin/env python3
"""
LTX-Video MCP Server
Verbindet Claude Code mit ComfyUI/LTX-Video für lokale Videogenerierung.
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_DIR = os.getenv("COMFYUI_DIR", str(Path.home() / "ComfyUI"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(Path(COMFYUI_DIR) / "output"))

server = Server("ltx-video")


def get_ltx_workflow(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    fps: int = 24,
    steps: int = 4,
    cfg: float = 1.0,
    seed: int = -1,
) -> dict:
    """Erstellt einen ComfyUI API-Workflow für LTX-Video."""
    if seed == -1:
        seed = int(time.time() * 1000) % (2**32)

    return {
        "1": {
            "class_type": "LTXVLoader",
            "inputs": {
                "ckpt_name": "ltx-video-2b-v0.9.7-distilled-04925-bf16.safetensors",
                "dtype": "bfloat16",
            },
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1],
            },
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1],
            },
        },
        "4": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        },
        "5": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["4", 0],
            },
        },
        "6": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler",
            },
        },
        "7": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["8", 0],
                "guider": ["9", 0],
                "sampler": ["6", 0],
                "sigmas": ["5", 0],
                "latent_image": ["4", 0],
            },
        },
        "8": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
        },
        "9": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["1", 0],
                "conditioning": ["2", 0],
            },
        },
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["1", 2],
            },
        },
        "11": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["10", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "ltx_video",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }


async def wait_for_comfyui(timeout: int = 30) -> bool:
    """Wartet bis ComfyUI erreichbar ist."""
    async with httpx.AsyncClient() as client:
        for _ in range(timeout):
            try:
                r = await client.get(f"{COMFYUI_URL}/system_stats", timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1)
    return False


async def submit_workflow(workflow: dict) -> str:
    """Schickt den Workflow an ComfyUI und gibt die prompt_id zurück."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["prompt_id"]


async def wait_for_result(prompt_id: str, timeout: int = 600) -> dict | None:
    """Wartet auf das Ergebnis und gibt die Output-Dateien zurück."""
    async with httpx.AsyncClient() as client:
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = await client.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
                history = r.json()
                if prompt_id in history:
                    entry = history[prompt_id]
                    if entry.get("status", {}).get("completed"):
                        return entry.get("outputs", {})
            except Exception:
                pass
            await asyncio.sleep(3)
    return None


def find_latest_video() -> str | None:
    """Findet das zuletzt generierte Video in ComfyUI output."""
    output = Path(OUTPUT_DIR)
    videos = list(output.glob("ltx_video*.mp4"))
    if not videos:
        return None
    return str(max(videos, key=lambda p: p.stat().st_mtime))


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_video",
            description=(
                "Generiert ein Video lokal mit LTX-Video via ComfyUI. "
                "Kostenlos, läuft vollständig auf dem lokalen iMac. "
                "Gibt den Pfad zum fertigen MP4 zurück."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Beschreibung des Videos (englisch für beste Ergebnisse)",
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Was im Video NICHT sein soll",
                        "default": "worst quality, inconsistent motion, blurry, jittery, distorted",
                    },
                    "width": {
                        "type": "integer",
                        "description": "Breite in Pixeln (muss durch 32 teilbar sein)",
                        "default": 768,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Höhe in Pixeln (muss durch 32 teilbar sein)",
                        "default": 512,
                    },
                    "num_frames": {
                        "type": "integer",
                        "description": "Anzahl der Frames (97 = ~4 Sekunden bei 24fps)",
                        "default": 97,
                    },
                    "fps": {
                        "type": "integer",
                        "description": "Frames pro Sekunde",
                        "default": 24,
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Sampling-Schritte (4 für distilled, 25-40 für full)",
                        "default": 4,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Seed für Reproduzierbarkeit (-1 = zufällig)",
                        "default": -1,
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="check_comfyui_status",
            description="Prüft ob ComfyUI läuft und bereit ist.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="start_comfyui",
            description="Startet den ComfyUI Server falls er nicht läuft.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_generated_videos",
            description="Listet alle bisher generierten Videos auf.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:

    if name == "check_comfyui_status":
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{COMFYUI_URL}/system_stats", timeout=5)
                data = r.json()
                return [TextContent(type="text", text=f"ComfyUI läuft. System: {json.dumps(data, indent=2)}")]
        except Exception as e:
            return [TextContent(type="text", text=f"ComfyUI nicht erreichbar: {e}\nStarte mit 'start_comfyui' Tool.")]

    elif name == "start_comfyui":
        venv_python = str(Path(COMFYUI_DIR) / ".venv" / "bin" / "python3")
        main_py = str(Path(COMFYUI_DIR) / "main.py")
        log_file = str(Path(COMFYUI_DIR) / "comfyui.log")

        # Prüfen ob bereits läuft
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{COMFYUI_URL}/system_stats", timeout=3)
                if r.status_code == 200:
                    return [TextContent(type="text", text="ComfyUI läuft bereits.")]
        except Exception:
            pass

        # Starten
        with open(log_file, "w") as log:
            subprocess.Popen(
                [venv_python, main_py, "--cpu", "--port", "8188"],
                cwd=COMFYUI_DIR,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

        # Warten bis erreichbar
        ready = await wait_for_comfyui(60)
        if ready:
            return [TextContent(type="text", text=f"ComfyUI erfolgreich gestartet auf {COMFYUI_URL}")]
        else:
            return [TextContent(type="text", text=f"ComfyUI gestartet, wartet noch. Log: {log_file}")]

    elif name == "list_generated_videos":
        output = Path(OUTPUT_DIR)
        videos = sorted(output.glob("ltx_video*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not videos:
            return [TextContent(type="text", text="Noch keine Videos generiert.")]
        lines = [f"{i+1}. {v.name} ({v.stat().st_size // 1024 // 1024}MB) — {v}" for i, v in enumerate(videos[:10])]
        return [TextContent(type="text", text="Generierte Videos:\n" + "\n".join(lines))]

    elif name == "generate_video":
        prompt = arguments["prompt"]
        negative_prompt = arguments.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted")
        width = arguments.get("width", 768)
        height = arguments.get("height", 512)
        num_frames = arguments.get("num_frames", 97)
        fps = arguments.get("fps", 24)
        steps = arguments.get("steps", 4)
        seed = arguments.get("seed", -1)

        # 1. ComfyUI erreichbar?
        try:
            async with httpx.AsyncClient() as client:
                await client.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        except Exception:
            return [TextContent(type="text", text=(
                "ComfyUI läuft nicht. Starte zuerst mit dem 'start_comfyui' Tool."
            ))]

        # 2. Workflow bauen & abschicken
        workflow = get_ltx_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            seed=seed,
        )

        try:
            prompt_id = await submit_workflow(workflow)
        except Exception as e:
            return [TextContent(type="text", text=f"Fehler beim Einreichen: {e}")]

        # 3. Auf Ergebnis warten
        result = await wait_for_result(prompt_id, timeout=600)

        if result is None:
            return [TextContent(type="text", text=f"Timeout — Video-Generierung dauert zu lange. Prompt-ID: {prompt_id}")]

        # 4. Video-Pfad ermitteln
        video_path = find_latest_video()

        if video_path:
            size_mb = Path(video_path).stat().st_size // 1024 // 1024
            return [TextContent(type="text", text=(
                f"Video erfolgreich generiert!\n"
                f"Prompt: {prompt}\n"
                f"Pfad: {video_path}\n"
                f"Größe: {size_mb}MB\n"
                f"Auflösung: {width}x{height} @ {fps}fps\n"
                f"Frames: {num_frames}\n"
                f"Seed: {seed}"
            ))]
        else:
            return [TextContent(type="text", text=f"Generierung abgeschlossen aber Video-Datei nicht gefunden. Outputs: {result}")]

    return [TextContent(type="text", text=f"Unbekanntes Tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
