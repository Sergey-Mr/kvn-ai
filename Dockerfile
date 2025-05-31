FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p ./models

# Fix line endings and make start script executable
RUN dos2unix start_server.sh && chmod +x start_server.sh

# Expose port
EXPOSE 8000

# Run the start script using bash explicitly
CMD ["bash", "./start_server.sh"]