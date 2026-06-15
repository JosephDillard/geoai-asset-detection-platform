FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision \
    && python -m pip install -e ".[dev,keras,pytorch]"

COPY config ./config
COPY scripts ./scripts
COPY sql ./sql

EXPOSE 8000

ENTRYPOINT ["python", "scripts/docker_entrypoint.py"]
CMD ["python", "-m", "geoai_roads.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
