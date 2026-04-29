#!/usr/bin/env python3
"""
LTX-Video MCP Server
Verbindet Claude Code mit ComfyUI/LTX-Video für lokale Videogenerierung.

Stack:
- ComfyUI via Docker (Linux + PyTorch 2.4 CPU)
- ltxv-2b-0.9.8-distilled-fp8.safetensors (Transformer + VAE)
- t5xxl_fp8_e4m3fn.safetensors (T5-Text-Encoder, separat geladen)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
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
COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", str(Path.home() / "ComfyUI")))
DOCKER_DIR = COMFYUI_DIR / "docker"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(COMFYUI_DIR / "output")))

server = Server("ltx-video")


def get_ltx_workflow(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    fps: int = 24,
    steps: int = 4,
    seed: int = -1,
) -> dict:
    """
    Erstellt den ComfyUI API-Workflow für LTX-Video 0.9.8.

    Knoten-Struktur:
      1  CheckpointLoaderSimple  → model[0], clip[1], vae[2]
      2  CLIPLoader (T5-xxl)     → clip[0]
      3  CLIPTextEncode positive → conditioning[0]
      4  CLIPTextEncode negative → conditioning[0]
      5  EmptyLTXVLatentVideo    → latent[0]
      6  LTXVConditioning        → pos[0], neg[1], latent[2]
      7  LTXVScheduler           → sigmas[0]
      8  KSamplerSelect          → sampler[0]
      9  RandomNoise             → noise[0]
     10  BasicGuider             → guider[0]
     11  SamplerCustomAdvanced   → samples[0]
     12  VAEDecode               → image[0]
     13  VHS_VideoCombine        → (speichert MP4)
    """
    if seed == -1:
        seed = int(time.time() * 1000) % (2**32)

    return {
        # Modell + VAE laden (kein T5 im Checkpoint enthalten)
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "ltxv-2b-0.9.8-distilled-fp8.safetensors",
            },
        },
        # T5-xxl Textencoder separat laden
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "ltxv",
            },
        },
        # Positives Prompt-Encoding
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["2", 0],
            },
        },
        # Negatives Prompt-Encoding
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["2", 0],
            },
        },
        # Leere Latent-Video-Maske erzeugen
        "5": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        },
        # LTX-Video Conditioning (fügt frame_rate-Metadaten hinzu)
        # Outputs: positive[0], negative[1]
        "6": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "frame_rate": fps,
            },
        },
        # Scheduler für distilled Modell (wenig Steps)
        # latent kommt von EmptyLTXVLatentVideo (Node 5)
        "7": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["5", 0],
            },
        },
        # Euler Sampler (Standard für LTX-Video)
        "8": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler",
            },
        },
        # Rauschen mit festem Seed
        "9": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
        },
        # BasicGuider: kein CFG (distilled → cfg=1.0 implizit)
        "10": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["1", 0],
                "conditioning": ["6", 0],
            },
        },
        # Sampling-Hauptknoten
        "11": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["10", 0],
                "sampler": ["8", 0],
                "sigmas": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        # Latents → Pixel-Frames dekodieren
        "12": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["11", 0],
                "vae": ["1", 2],
            },
        },
        # Frames → MP4 zusammenbauen (VideoHelperSuite)
        "13": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["12", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "filename_prefix": "ltx_video",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
            },
        },
    }


async def wait_for_comfyui(timeout: int = 120) -> bool:
    """Wartet bis ComfyUI auf Port 8188 antwortet."""
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


async def wait_for_result(prompt_id: str, timeout: int = 1800) -> dict | None:
    """
    Wartet auf das Ergebnis und gibt die Output-Dateien zurück.
    Timeout 1800s (30 Min) für CPU-Only Generierung.
    """
    async with httpx.AsyncClient() as client:
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = await client.get(
                    f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
                )
                history = r.json()
                if prompt_id in history:
                    entry = history[prompt_id]
                    if entry.get("status", {}).get("completed"):
                        return entry.get("outputs", {})
            except Exception:
                pass
            await asyncio.sleep(5)
    return None


def find_latest_video() -> str | None:
    """Findet die zuletzt generierte Video-Datei (MP4 oder WebP) im output-Ordner."""
    candidates = []
    for pattern in ["ltx_video*.mp4", "ltx_video*.webp", "ltx_video*.webm"]:
        candidates.extend(OUTPUT_DIR.glob(pattern))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_mtime))


def docker_compose_cmd() -> list[str]:
    """Gibt den Docker-Compose-Befehl zurück."""
    return ["docker", "compose", "-f", str(DOCKER_DIR / "docker-compose.yml")]


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_video",
            description=(
                "Generiert ein Video lokal mit LTX-Video 0.9.8 via ComfyUI (Docker). "
                "Kostenlos, läuft vollständig auf dem lokalen iMac. "
                "Gibt den Pfad zum fertigen MP4 zurück. "
                "WICHTIG: ComfyUI muss laufen (start_comfyui). "
                "CPU-Generierung: 97 Frames ≈ 15-40 Min."
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
                        "description": "Breite in Pixeln (muss durch 32 teilbar sein, empfohlen: 512-768)",
                        "default": 512,
                    },
                    "height": {
                        "type": "integer",
                        "description": "Höhe in Pixeln (muss durch 32 teilbar sein, empfohlen: 320-512)",
                        "default": 320,
                    },
                    "num_frames": {
                        "type": "integer",
                        "description": "Anzahl der Frames (25 = ~1s bei 24fps, schneller zum Testen; 97 = ~4s)",
                        "default": 25,
                    },
                    "fps": {
                        "type": "integer",
                        "description": "Frames pro Sekunde",
                        "default": 24,
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Sampling-Schritte (4-8 für distilled Modell)",
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
            description="Prüft ob ComfyUI (Docker) läuft und bereit ist.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="start_comfyui",
            description=(
                "Startet den ComfyUI Docker-Container. "
                "VORAUSSETZUNG: Docker Desktop muss laufen und mindestens 40 GB RAM zugewiesen haben "
                "(Docker Desktop → Settings → Resources → Memory)."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="stop_comfyui",
            description="Stoppt den ComfyUI Docker-Container.",
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
        # Docker-Container-Status
        try:
            result = subprocess.run(
                docker_compose_cmd() + ["ps", "--format", "json"],
                capture_output=True, text=True, timeout=10,
            )
            container_info = result.stdout.strip()
        except Exception as e:
            container_info = f"Docker-Fehler: {e}"

        # HTTP-Erreichbarkeit
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{COMFYUI_URL}/system_stats", timeout=5)
                data = r.json()
                return [TextContent(type="text", text=(
                    f"ComfyUI läuft auf {COMFYUI_URL}\n"
                    f"System: {json.dumps(data, indent=2)}\n"
                    f"Container: {container_info}"
                ))]
        except Exception as e:
            return [TextContent(type="text", text=(
                f"ComfyUI nicht erreichbar: {e}\n"
                f"Container-Status: {container_info}\n"
                f"→ Starte mit 'start_comfyui' Tool."
            ))]

    elif name == "start_comfyui":
        # Bereits laufend?
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{COMFYUI_URL}/system_stats", timeout=3)
                if r.status_code == 200:
                    return [TextContent(type="text", text="ComfyUI läuft bereits.")]
        except Exception:
            pass

        # Docker Desktop laufend?
        try:
            result = subprocess.run(
                ["docker", "info"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return [TextContent(type="text", text=(
                    "Docker Desktop ist nicht gestartet.\n"
                    "→ Docker Desktop öffnen und starten, dann erneut versuchen."
                ))]
        except FileNotFoundError:
            return [TextContent(type="text", text=(
                "Docker nicht gefunden. Docker Desktop installieren: https://docker.com"
            ))]

        # Container starten
        try:
            result = subprocess.run(
                docker_compose_cmd() + ["up", "-d"],
                capture_output=True, text=True, timeout=120,
                cwd=str(DOCKER_DIR),
            )
            if result.returncode != 0:
                return [TextContent(type="text", text=(
                    f"Docker-Start fehlgeschlagen:\n{result.stderr}\n\n"
                    f"WICHTIG: Docker Desktop → Settings → Resources → Memory auf min. 40 GB setzen!"
                ))]
        except Exception as e:
            return [TextContent(type="text", text=f"Fehler beim Docker-Start: {e}")]

        # Auf ComfyUI warten (bis zu 2 Minuten für erstes Laden)
        ready = await wait_for_comfyui(120)
        if ready:
            return [TextContent(type="text", text=(
                f"ComfyUI erfolgreich gestartet auf {COMFYUI_URL}\n"
                f"Docker-Container läuft in {DOCKER_DIR}"
            ))]
        else:
            return [TextContent(type="text", text=(
                f"Docker gestartet, ComfyUI antwortet noch nicht.\n"
                f"Logs: docker compose -f {DOCKER_DIR}/docker-compose.yml logs -f\n"
                f"HINWEIS: Docker Desktop braucht mind. 40 GB RAM-Zuweisung!"
            ))]

    elif name == "stop_comfyui":
        try:
            result = subprocess.run(
                docker_compose_cmd() + ["down"],
                capture_output=True, text=True, timeout=30,
                cwd=str(DOCKER_DIR),
            )
            if result.returncode == 0:
                return [TextContent(type="text", text="ComfyUI Docker-Container gestoppt.")]
            else:
                return [TextContent(type="text", text=f"Fehler beim Stoppen:\n{result.stderr}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Fehler: {e}")]

    elif name == "list_generated_videos":
        candidates = []
        for pattern in ["ltx_video*.mp4", "ltx_video*.webp", "ltx_video*.webm"]:
            candidates.extend(OUTPUT_DIR.glob(pattern))
        videos = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        if not videos:
            return [TextContent(type="text", text="Noch keine Videos generiert.")]
        lines = [
            f"{i+1}. {v.name} ({v.stat().st_size // 1024 // 1024} MB) — {v}"
            for i, v in enumerate(videos[:10])
        ]
        return [TextContent(type="text", text="Generierte Videos:\n" + "\n".join(lines))]

    elif name == "generate_video":
        prompt = arguments["prompt"]
        negative_prompt = arguments.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted",
        )
        width = arguments.get("width", 512)
        height = arguments.get("height", 320)
        num_frames = arguments.get("num_frames", 25)
        fps = arguments.get("fps", 24)
        steps = arguments.get("steps", 4)
        seed = arguments.get("seed", -1)

        # 1. ComfyUI erreichbar?
        try:
            async with httpx.AsyncClient() as client:
                await client.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        except Exception:
            return [TextContent(type="text", text=(
                "ComfyUI läuft nicht.\n"
                "1. Docker Desktop starten\n"
                "2. Docker Desktop → Settings → Resources → Memory: 40 GB setzen\n"
                "3. 'start_comfyui' Tool aufrufen"
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

        eta_min = max(5, (num_frames * steps) // 4)
        return_msg = (
            f"Workflow eingereicht! Prompt-ID: {prompt_id}\n"
            f"Prompt: {prompt[:100]}\n"
            f"Auflösung: {width}x{height} | {num_frames} Frames @ {fps}fps | {steps} Steps\n"
            f"Seed: {seed}\n"
            f"Geschätzte Zeit (CPU): ~{eta_min}-{eta_min*3} Minuten\n"
            f"Warte auf Ergebnis..."
        )

        # 3. Auf Ergebnis warten (lange Timeout für CPU)
        result = await wait_for_result(prompt_id, timeout=1800)

        if result is None:
            return [TextContent(type="text", text=(
                f"Timeout nach 30 Minuten.\n"
                f"Prompt-ID: {prompt_id}\n"
                f"ComfyUI läuft noch unter {COMFYUI_URL} — Video wird evtl. noch generiert."
            ))]

        # 4. Video-Pfad ermitteln
        video_path = find_latest_video()

        if video_path:
            size_mb = Path(video_path).stat().st_size // 1024 // 1024
            return [TextContent(type="text", text=(
                f"Video erfolgreich generiert!\n"
                f"Pfad: {video_path}\n"
                f"Größe: {size_mb} MB\n"
                f"Prompt: {prompt}\n"
                f"Auflösung: {width}x{height} @ {fps}fps | {num_frames} Frames\n"
                f"Seed: {seed}"
            ))]
        else:
            return [TextContent(type="text", text=(
                f"Generierung abgeschlossen, aber Video-Datei nicht gefunden.\n"
                f"Outputs: {json.dumps(result, indent=2)}\n"
                f"Output-Ordner: {OUTPUT_DIR}"
            ))]

    return [TextContent(type="text", text=f"Unbekanntes Tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
