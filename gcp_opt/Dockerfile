# 1. Upgraded PyTorch image with CUDA support
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Set working directory
WORKDIR /app

# 2. Pin transformers, keep tiktoken, drop JAX/Diffrax
RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    pandas \
    accelerate \
    tiktoken \
    google-cloud-storage

# Copy the optimization script
COPY cloud_gcg_optimizer.py /app/cloud_gcg_optimizer.py

# Create directories for mounting volumes
RUN mkdir -p /app/data /app/output

# 3. The critical execution command
ENTRYPOINT ["python", "-u", "/app/cloud_gcg_optimizer.py"]