# Use NVIDIA CUDA base image supporting CUDA 12.1
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install Python 3.12, pip, and necessary build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    python3.12-venv \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

# Update pip and set python3.12 as default python3
RUN python3.12 -m pip install --upgrade pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Set the working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.1 support
# Check https://pytorch.org/get-started/locally/ for the latest command if needed
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Download NLTK data during build
# This avoids potential runtime download issues and SSL errors
RUN python3 -m nltk.downloader -d /usr/share/nltk_data punkt

# Copy the rest of the application code
COPY app.py .
# Note: Consider using volumes for uploads and vector_db in production
# COPY uploads/ ./uploads/
# COPY vector_db/ ./vector_db/

# Expose the port the app runs on (changed to 8080)
EXPOSE 8080

# Command to run the application using Uvicorn on port 8080
# Use 0.0.0.0 to make it accessible outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
