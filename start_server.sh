set -e

echo "Starting KVN AI Embedding Server..."

# Check if models directory exists and has content
if [ ! -d "./models" ] || [ -z "$(ls -A ./models 2>/dev/null)" ]; then
    echo "Models directory is empty or doesn't exist. Downloading embedding model..."
    python3 embedding-modal-downloader.py
    echo "Model download completed."
else
    echo "Models directory found with content. Skipping download."
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Start the server
echo "Starting FastAPI server..."
python3 pinecone_server.py