from sentence_transformers import SentenceTransformer

model = SentenceTransformer("thenlper/gte-small")
model.save('./models/gte-small') 
