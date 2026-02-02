# Dockerfile for VLA Pipeline
# Provides reproducible containerized environment

FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip==23.3.1 && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Set deterministic hash seed
ENV PYTHONHASHSEED=42

# Default command
CMD ["python", "demo/demo_basic.py", "--seed", "42", "--headless"]
