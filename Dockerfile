# Use official PyTorch runtime with CUDA 12.1 as base image
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set workspace directory
WORKDIR /app

# Install system dependencies (git is required for GitPython to track repositories)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependency specifications
COPY requirements.txt /app/

# Install Python requirements
# Pre-installed pytorch in the base image is utilized, --extra-index-url is provided for resolving any CUDA-specific torch modules
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy project files
COPY api/ /app/api/
COPY src/ /app/src/
COPY main.py /app/main.py

# Expose backend port
EXPOSE 8000

# Start uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
