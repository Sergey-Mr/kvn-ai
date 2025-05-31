from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone
import os
import uvicorn
from typing import List, Optional

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

class SearchRequest(BaseModel):
    text: str
    k: Optional[int] = 5
    include_metadata: Optional[bool] = True

class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Optional[dict] = None

class SearchResponse(BaseModel):
    status: str
    query: str
    results: List[SearchResult]

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

@app.post("/search", response_model=SearchResponse)
def search_similar(request: SearchRequest):
    # Convert user text to embedding
    query_embedding = model.encode(request.text).tolist()
    
    # Perform k-NN search in Pinecone
    search_results = index.query(
        vector=query_embedding,
        top_k=request.k,
        include_metadata=request.include_metadata
    )
    
    # Format results
    results = []
    for match in search_results['matches']:
        result = SearchResult(
            id=match['id'],
            score=match['score'],
            metadata=match.get('metadata') if request.include_metadata else None
        )
        results.append(result)
    
    return SearchResponse(
        status="success",
        query=request.text,
        results=results
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)