# ==============================================================================
# Multi-stage Dockerfile for BaBank Customer Retention Platform
# Stage 1: Build modern React 19 Frontend (Vite)
# Stage 2: Production Python 3.11 container serving both API & UI on a single port
# ==============================================================================

# Stage 1: Build Frontend Assets
FROM node:22-alpine AS frontend-builder
WORKDIR /build/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Stage 2: Python Backend & Unified Production Server
FROM python:3.11-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, models, configuration, and data
COPY api/ ./api/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/
COPY results/ ./results/
COPY preprocessing/ ./preprocessing/
COPY utils/ ./utils/
COPY api.py main.py ./

# Copy built frontend from Stage 1 into frontend/dist
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

EXPOSE 8000

# Start unified FastAPI web service (serves both React UI and REST API)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
