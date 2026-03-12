# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for opuslib and rnnoise
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopus0 \
    libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# --- Dependency layer (cached unless pyproject.toml changes) ---
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install $(python -c "\
import tomllib, pathlib; \
d = tomllib.loads(pathlib.Path('pyproject.toml').read_text()); \
deps = d['project']['dependencies'] + d['project'].get('optional-dependencies',{}).get('audio',[]); \
print(' '.join(deps))")

# --- Source layer (rebuilds when code changes) ---
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps ".[audio]"

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
