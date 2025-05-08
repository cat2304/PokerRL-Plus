FROM --platform=linux/amd64 python:3.6-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    requests \
    torch==0.4.1 \
    numpy \
    scipy

# Install PokerRL
RUN pip install --no-cache-dir PokerRL

# Copy the project files
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["bash"] 