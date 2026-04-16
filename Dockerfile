FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}"

WORKDIR /app

RUN set -eux; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get clean; \
    for i in 1 2 3; do \
        apt-get update && break || sleep 10; \
    done; \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        ca-certificates \
        build-essential \
        libjpeg-dev \
        zlib1g-dev; \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --python python3

COPY . .

CMD ["uv", "run", "--python", "python3", "main.py"]