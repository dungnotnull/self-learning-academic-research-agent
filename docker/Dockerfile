FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -m -u 1000 agentuser && chown -R agentuser:agentuser /app
USER agentuser

# Copy application code
COPY --chown=agentuser:agentuser . .

# Create data and models directories
RUN mkdir -p /app/data /app/models

# Expose REST API port
EXPOSE 8018

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8018/health || exit 1

# Default command: start the REST server with daily scheduler
CMD ["python", "-m", "agent.main", "serve", "--start-scheduler"]
