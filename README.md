# LTX-Video MCP Server

Verbindet Claude Code mit LTX-Video via ComfyUI für **kostenlose, lokale Videogenerierung**.

## Setup

### Voraussetzungen
- macOS (Intel oder Apple Silicon)
- Python 3.12+
- ComfyUI installiert unter `~/ComfyUI`
- LTX-Video Modell unter `~/ComfyUI/models/video_models/`

### Installation

```bash
cd ~/ComfyUI/ltx-mcp
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install mcp httpx
```

### MCP-Server registrieren

```bash
claude mcp add ltx-video --scope user \
  /Users/samsimac/ComfyUI/ltx-mcp/.venv/bin/python3 \
  /Users/samsimac/ComfyUI/ltx-mcp/server.py \
  -e COMFYUI_DIR=/Users/samsimac/ComfyUI \
  -e COMFYUI_URL=http://127.0.0.1:8188
```

## Verwendung

### ComfyUI starten
```bash
bash ~/ComfyUI/ltx-mcp/start_comfyui.sh
```

### Video generieren (via Claude Code)
Einfach in Claude Code sagen:
> "Generiere ein Video von einem Sonnenuntergang am Strand"

Claude startet automatisch ComfyUI und generiert das Video lokal.

## MCP Tools

| Tool | Beschreibung |
|------|-------------|
| `generate_video` | Video per Text-Prompt generieren |
| `start_comfyui` | ComfyUI Server starten |
| `check_comfyui_status` | Server-Status prüfen |
| `list_generated_videos` | Alle generierten Videos auflisten |

## Modell

**LTX-Video 0.9.7 Distilled** von Lightricks
- Schneller als das volle Modell (4 Steps statt 25-40)
- ~5GB Dateigröße
- 768x512px Standard, bis 1280x720px möglich

## Hardware

Getestet auf iMac (Intel) mit AMD Radeon Pro 5700 XT (16GB VRAM), 64GB RAM.
Läuft im CPU-Modus — Generierung dauert ca. 5-15 Min pro Video.
