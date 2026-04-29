#!/bin/bash
# ComfyUI mit LTX-Video starten (CPU-Modus für Intel/AMD iMac)
COMFYUI_DIR="$HOME/ComfyUI"
VENV="$COMFYUI_DIR/.venv/bin/python3"
LOG="$COMFYUI_DIR/comfyui.log"

echo "Starte ComfyUI..."
cd "$COMFYUI_DIR"
nohup "$VENV" main.py --cpu --port 8188 > "$LOG" 2>&1 &
echo "ComfyUI PID: $! — Log: $LOG"
echo "Warte auf Start..."
sleep 5
curl -s http://127.0.0.1:8188/system_stats | python3 -m json.tool 2>/dev/null && echo "ComfyUI bereit!" || echo "Noch nicht bereit, warte etwas länger..."
