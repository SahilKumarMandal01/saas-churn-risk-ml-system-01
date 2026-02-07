# ============================================================
# Base Image
# ============================================================
# Slim image for smaller size, faster pulls, lower attack surface
FROM python:3.12-slim

# ============================================================
# Environment Variables
# ============================================================
# - Prevents Python from writing .pyc files
# - Ensures logs are flushed immediately (important for containers)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ============================================================
# Set Working Directory
# ============================================================
WORKDIR /app

# ============================================================
# System Dependencies (Minimal)
# ============================================================
# build-essential is required for some Python wheels (numpy, sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Python Dependencies (Inference Only)
# ============================================================
# Copy only inference requirements to leverage Docker layer caching
COPY requirements/ requirements/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements/inference.txt

# ============================================================
# Copy Inference Code Only (Explicit & Intentional)
# ============================================================
COPY src/api src/api
COPY src/pipeline/prediction_pipeline.py src/pipeline/prediction_pipeline.py
COPY src/exception.py src/exception.py
COPY src/logging.py src/logging.py
COPY src/utils src/utils
COPY src/constants src/constants


# ============================================================
# Copy Final Model Artifacts
# ============================================================
COPY final_model final_model

# ============================================================
# Expose API Port
# ============================================================
EXPOSE 8000

# ============================================================
# Run FastAPI App
# ============================================================
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]