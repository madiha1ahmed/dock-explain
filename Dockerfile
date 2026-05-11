FROM mambaorg/micromamba:1.5.10

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLBACKEND=Agg
ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODELS=/app/ollama_models

COPY environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*
    
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY . /app

RUN mkdir -p /app/data/results /app/ollama_models && \
    chown -R mambauser:mambauser /app/data /app/ollama_models /app && \
    chmod +x /app/start.sh

USER mambauser

EXPOSE 8000

CMD ["bash", "-lc", "/app/start.sh"]
