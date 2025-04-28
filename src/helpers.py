from sentence_transformers import SentenceTransformer
import os

# Class for Hugging Face embeddings
class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
        
    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Download the embedding model from Hugging Face
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings