# LTX-Video MCP Server

Verbindet Claude Code mit **LTX-Video 0.9.8** via ComfyUI (Docker) für kostenlose, lokale Videogenerierung.

## Architektur

```
Claude Code
    ↓ MCP
server.py  (MCP Server)
    ↓ HTTP :8188
ComfyUI (Docker Container)
    ↓ ComfyUI Workflow
ltxv-2b-0.9.8-distilled-fp8.safetensors  +  t5xxl_fp8_e4m3fn.safetensors
    ↓
MP4 → ~/ComfyUI/output/
```

## Voraussetzungen

- macOS (Intel Core i9 / Apple Silicon)
- Docker Desktop ≥ 4.x
- **Docker Desktop RAM: mindestens 40 GB** (Settings → Resources → Memory)
- Modelle heruntergeladen (s.u.)

## Modelle

| Datei | Pfad | Größe | Quelle |
|-------|------|-------|--------|
| `ltxv-2b-0.9.8-distilled-fp8.safetensors` | `~/ComfyUI/models/checkpoints/` | 4,2 GB | [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) |
| `t5xxl_fp8_e4m3fn.safetensors` | `~/ComfyUI/models/text_encoders/` | 4,6 GB | [comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) |

Nach dem Download:
```bash
# Symlink damit CLIPLoader die T5-Datei findet
ln -s ../text_encoders/t5xxl_fp8_e4m3fn.safetensors \
      ~/ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors
```

## Setup

### 1. Docker-Image bauen

```bash
cd ~/ComfyUI/docker
docker compose build
```

> Einmalig, dauert ~10 Min (PyTorch + ComfyUI + Custom Nodes).

### 2. MCP-Server installieren

```bash
cd ~/ComfyUI/ltx-mcp
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install mcp httpx
```

### 3. MCP-Server bei Claude Code registrieren

```bash
claude mcp add ltx-video --scope user \
  /Users/samsimac/ComfyUI/ltx-mcp/.venv/bin/python3 \
  /Users/samsimac/ComfyUI/ltx-mcp/server.py \
  -e COMFYUI_DIR=/Users/samsimac/ComfyUI \
  -e COMFYUI_URL=http://127.0.0.1:8188
```

## Verwendung

### Option A: Direkt in Claude Code

```
"Starte ComfyUI und generiere ein Video von einem Sonnenuntergang am Strand"
```

Claude Code nutzt automatisch die MCP-Tools.

### Option B: Manuell

```bash
# Docker Desktop starten, dann:
docker compose -f ~/ComfyUI/docker/docker-compose.yml up -d

# Warten bis ComfyUI bereit ist (~60s beim ersten Mal)
curl http://localhost:8188/system_stats
```

## MCP Tools

| Tool | Beschreibung |
|------|-------------|
| `generate_video` | Video per Text-Prompt generieren |
| `start_comfyui` | ComfyUI Docker-Container starten |
| `stop_comfyui` | ComfyUI Docker-Container stoppen |
| `check_comfyui_status` | Server-Status prüfen |
| `list_generated_videos` | Alle generierten Videos auflisten |

## Parameter

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `prompt` | — | Text-Beschreibung (englisch empfohlen) |
| `negative_prompt` | `worst quality...` | Was nicht im Video sein soll |
| `width` | 512 | Breite (Vielfaches von 32) |
| `height` | 320 | Höhe (Vielfaches von 32) |
| `num_frames` | 25 | Frames (25 ≈ 1s, 97 ≈ 4s) |
| `fps` | 24 | Frames pro Sekunde |
| `steps` | 4 | Sampling-Steps (4–8 für distilled) |
| `seed` | -1 | Seed (-1 = zufällig) |

## ComfyUI Workflow (intern)

```
CheckpointLoaderSimple (ltxv-2b-0.9.8)
    → model, vae
CLIPLoader (t5xxl_fp8_e4m3fn, type=ltxv)
    → text encoder
CLIPTextEncode (positive + negative)
EmptyLTXVLatentVideo
LTXVConditioning (frame_rate)
LTXVScheduler (steps=4, distilled-optimiert)
KSamplerSelect (euler) + RandomNoise + BasicGuider
SamplerCustomAdvanced
VAEDecode
VHS_VideoCombine → MP4
```

## Geschätzte Laufzeiten (CPU-Only)

| Auflösung | Frames | Steps | Zeit |
|-----------|--------|-------|------|
| 512×320 | 25 | 4 | ~5–10 Min |
| 512×320 | 97 | 4 | ~20–40 Min |
| 768×512 | 97 | 4 | ~60–120 Min |

## Hardware (getestet)

- iMac 2019 (Intel Core i9, 64 GB RAM, AMD Radeon Pro 5700 XT)
- Docker Desktop mit 40+ GB RAM-Zuweisung
- CPU-Inferenz (kein CUDA/Metal erforderlich)

## Warum Docker?

LTX-Video 0.9.8 nutzt einen neuen VAE-Aufbau (`LTXVideo095DownBlock3D`), der erst ab `diffusers 0.34+` unterstützt wird. diffusers 0.34+ benötigt PyTorch 2.4+, für das auf Intel-Mac (x86_64) keine nativen Wheels existieren. Docker löst das durch Linux-Container mit PyTorch 2.4 CPU.
