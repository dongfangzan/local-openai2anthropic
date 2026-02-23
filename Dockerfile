# SPDX-License-Identifier: Apache-2.0
# local-openai2anthropic Docker Image

FROM python:3.12-slim

# Build arguments for metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=0.4.0

# Labels
LABEL org.opencontainers.image.title="local-openai2anthropic" \
      org.opencontainers.image.description="A lightweight proxy server that converts Anthropic Messages API to OpenAI API" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/dongfangzan/local-openai2anthropic" \
      org.opencontainers.image.license="Apache-2.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md .
COPY src ./src

# Install the package
RUN pip install --no-cache-dir .

# Create config directory
RUN mkdir -p /app/config

# Expose the default port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# Set the entrypoint
ENTRYPOINT ["oa2a"]
