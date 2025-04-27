from src.helpers import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data = load_pdf_file("data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

pc.create_index(
    name=index_name,
    dimension=384,  # 384 is the dimension of the all-MiniLM-L6-v2 model
    metric="cosine",
    spec = ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)     

#Store
#Embed each chunk and upsert the embeddings into the Pinecone index
docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks, # list of documents to be embedded
    index_name = index_name, # name of the index to upsert the embeddings into
    embedding = embeddings, # embedding model to use for embedding the documents
)