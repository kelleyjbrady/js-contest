# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install required Python packages
RUN pip install --no-cache-dir \
    transformers \
    pandas \
    accelerate \
    tiktoken \
    jax \
    jaxlib \
    diffrax

# Copy the optimization script
COPY cloud_gcg_optimizer.py /app/cloud_gcg_optimizer.py

# Create directories for mounting volumes
RUN mkdir -p /app/data /app/output

# Set the entrypoint to run the script
ENTRYPOINT ["python", "/app/cloud_gcg_optimizer.py"]