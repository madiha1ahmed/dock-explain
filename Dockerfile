FROM mambaorg/micromamba:1.5.10

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLBACKEND=Agg

COPY environment.yml /tmp/environment.yml

RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

COPY . /app

EXPOSE 8000

CMD ["bash", "-lc", "gunicorn web_app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 1 --timeout 1800"]
