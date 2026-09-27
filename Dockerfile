# Stage 1: Builder
FROM python:3.12 AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first for Docker caching
COPY pyproject.toml uv.lock ./

# Install production dependencies
RUN uv sync --frozen --no-dev --no-install-project


# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY src/ /app/src/

# Make virtual environment executables available
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]