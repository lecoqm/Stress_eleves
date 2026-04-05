FROM ubuntu:22.04

# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Install project dependencies
COPY pyproject.toml .
RUN uv sync

CMD ["uv", "run", "main.py"]
