from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader #for loading pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter  #for chunking
from sentence_transformers import SentenceTransformer

# Extract text from PDF files
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Split the data into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

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