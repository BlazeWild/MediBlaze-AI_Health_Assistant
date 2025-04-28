from sentence_transformers import SentenceTransformer
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Class for Hugging Face embeddings
class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        self.model = SentenceTransformer(model_name)
        elapsed_time = time.time() - start_time
        logger.info(f"Model loaded in {elapsed_time:.2f} seconds")
        
    def embed_documents(self, texts):
        logger.info(f"Embedding {len(texts)} documents")
        start_time = time.time()
        result = self.model.encode(texts).tolist()
        elapsed_time = time.time() - start_time
        logger.info(f"Documents embedded in {elapsed_time:.2f} seconds")
        return result
        
    def embed_query(self, text):
        logger.info(f"Embedding query: {text[:30]}...")
        start_time = time.time()
        result = self.model.encode(text).tolist()
        elapsed_time = time.time() - start_time
        logger.info(f"Query embedded in {elapsed_time:.2f} seconds")
        return result

# Download the embedding model from Hugging Face
def download_hugging_face_embeddings():
    logger.info("Initializing embeddings model")
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elapsed_time = time.time() - start_time
    logger.info(f"Embeddings model initialized in {elapsed_time:.2f} seconds")
    return embeddings