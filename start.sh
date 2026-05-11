#!/usr/bin/env bash
set -e

echo "=================================================="
echo "Starting DockExplain with Ollama + Gunicorn"
echo "=================================================="

export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/app/ollama_models

mkdir -p /app/data/results
mkdir -p /app/ollama_models

echo "[1/4] Starting Ollama server..."
ollama serve > /app/data/ollama.log 2>&1 &

echo "[2/4] Waiting for Ollama to become available..."
for i in {1..60}; do
  if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "✓ Ollama is running"
    break
  fi

  if [ "$i" -eq 60 ]; then
    echo "✗ Ollama did not start in time"
    echo "---- Ollama log ----"
    cat /app/data/ollama.log || true
    exit 1
  fi

  sleep 2
done

echo "[3/4] Ensuring Gemma model is available..."
ollama pull gemma4:e2b || {
  echo "✗ Failed to pull gemma4:e2b"
  echo "Trying fallback model name: gemma3:1b"
  ollama pull gemma3:1b || true
}

echo "Available Ollama models:"
ollama list || true

echo "[4/4] Starting Flask app with Gunicorn..."
exec micromamba run -n base gunicorn web_app:app \
  --bind 0.0.0.0:${PORT:-8000} \
  --workers 1 \
  --threads 1 \
  --timeout 1800