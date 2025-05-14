# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required by unstructured and potentially other libraries
# libmagic1 for file type detection, poppler-utils for PDF processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libmagic1 \
       poppler-utils \
    # Clean up APT when done
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# This might download HuggingFace models if specified implicitly by langchain/transformers
# Consider pre-downloading models in a separate step if startup time is critical
RUN pip install --timeout 180 --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the uploads directory (if it doesn't exist in the copy)
# This directory is used by the application to temporarily store uploaded files
RUN mkdir -p uploads

# Make port 8222 available to the world outside this container
EXPOSE 8222

# Define the command to run the application
# Use 0.0.0.0 to allow connections from outside the container
# Set workers based on CPU cores if needed for production (e.g., using gunicorn)
# For simplicity, using uvicorn directly here. reload=False for production.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8222"]
