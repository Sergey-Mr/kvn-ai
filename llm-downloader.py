from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

def is_model_downloaded(model_path: str) -> bool:
    if not os.path.exists(model_path):
        return False
    return len(os.listdir(model_path)) > 0

def download_embedding_model():
    model_path = './models/gte-small'
    if is_model_downloaded(model_path):
        print("Embedding model already exists, skipping download...")
        return
    
    print("Downloading embedding model...")
    model = SentenceTransformer("thenlper/gte-small")
    model.save(model_path)
    print("Embedding model downloaded successfully")

if __name__ == "__main__":
    download_embedding_model()