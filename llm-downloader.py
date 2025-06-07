from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
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

def download_phi_model():
    model_path = './models/phi'
    if is_model_downloaded(model_path):
        print("Phi model already exists, skipping download...")
        return
    
    print("Downloading Phi model...")
    model_name = "microsoft/phi-1" 
    
    os.makedirs(model_path, exist_ok=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        
        # Save both locally
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        print("Phi model downloaded successfully")
    except Exception as e:
        print(f"Error downloading Phi model: {str(e)}")
        print("Please make sure you're logged in to Hugging Face:")
        print("Run: huggingface-cli login")
        return

if __name__ == "__main__":
    download_embedding_model()
    download_phi_model()