# Base image (CPU only)
FROM python:3.13-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# System dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Working directory
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml ./

# Install dependencies (CPU only)
RUN uv sync --no-dev

# Copy application code
COPY src/ src/
COPY inference/ inference/
COPY artifacts/ artifacts/
COPY models/ models/
COPY logs/ logs/
COPY main.py .
COPY README.md .

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uv", "run", "uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
