FROM ubuntu:22.04

# Install Python
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Install project dependencies
COPY pyproject.toml .
RUN uv sync

COPY main.py .
COPY src ./src
CMD ["uv", "run", "main.py"]
