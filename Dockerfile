# =========================================================
# Stage 1: Build the React/Vite dashboard
# =========================================================
FROM node:20-alpine AS build-frontend
WORKDIR /app/dashboard

COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci

COPY dashboard/ ./

# Tell Vite to use relative paths (empty string) since API & UI run on the same port in Docker
ENV VITE_API_BASE_URL=""
RUN npm run build

# =========================================================
# Stage 2: Backend + Serve Frontend Static Assets
# =========================================================
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS final
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (git is required for GitPython)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy python dependencies
COPY requirements.txt /app/

# Install python packages
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy backend codebase
COPY api/ /app/api/
COPY src/ /app/src/
COPY main.py /app/main.py
COPY test_datasets/ /app/test_datasets/

# Copy the built React frontend static files from Stage 1 into /app/dist
COPY --from=build-frontend /app/dashboard/dist /app/dist

# Expose the single unified port
EXPOSE 6767

# Start uvicorn server on the unified port 6767
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "6767"]
