from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone
import os
import uvicorn

# Load environment variables
load_dotenv()
model_name = os.getenv("MODEL_NAME")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Load embedding model
model = SentenceTransformer('./models/gte-small', device='cpu')

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

app = FastAPI()

class EmbedRequest(BaseModel):
    id: str
    text: str

@app.post("/embed")
def embed_and_store(request: EmbedRequest):
    embedding = model.encode(request.text).tolist()

    metadata = {"text": request.text}

    index.upsert([
        (request.id, embedding, metadata)
    ])

    return {
        "status": "success",
        "id": request.id,
        "metadata": metadata
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)