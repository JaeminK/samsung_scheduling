# Use NVIDIA CUDA base image with PyTorch
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0,1
ENV NCCL_DEBUG=INFO
ENV NCCL_IB_DISABLE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create cache directory for models
RUN mkdir -p /workspace/cache

# Set default command
CMD ["/bin/bash"]
