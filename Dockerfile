FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System packages needed by common scientific Python libs + PyMOL/OpenBabel basics
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    openbabel \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /app/requirements.txt

COPY . /app

WORKDIR /app/python_scripts

EXPOSE 8000

CMD gunicorn web_app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 4 --timeout 900