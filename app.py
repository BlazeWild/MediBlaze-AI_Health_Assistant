from flask import Flask, render_template, jsonify, request, session
from src.helpers import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))  # For session management

load_dotenv()

# Get environment variables with error handling
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate required environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
#Load existing index from pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name, # name of the index to upsert the embeddings into
    embedding = embeddings, # embedding model to use for embedding the documents
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 10}) # k is the number of documents to retrieve

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
    google_api_key=GEMINI_API_KEY
)

# Create main RAG chain for medical queries
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("human", "{context}")
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Simple classifier to determine if this is casual conversation or requires medical information
classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Determine if this message is:
         1. A casual greeting or non-medical conversation (CASUAL)
         2. A message that requires medical knowledge to answer (MEDICAL)
         Respond with only one word: either CASUAL or MEDICAL."""),
        ("human", "{input}")
    ]
)

@app.route("/")
def index():
    # Reset conversation history when loading the main page
    session.clear()
    current_time = datetime.now().strftime("%H:%M")
    return render_template("index.html", current_time=current_time)

@app.route("/get", methods=["GET","POST"])
def chat():
    try:
        msg = request.form.get("msg")
        if not msg:
            return "Please ask a question", 400
            
        # Initialize conversation history if it doesn't exist
        if 'conversation_history' not in session:
            session['conversation_history'] = []
            
        # Add user message to history
        session['conversation_history'].append({"role": "user", "content": msg})
        
        # Get conversation context (last few exchanges)
        conversation_context = session['conversation_history'][-6:]
        
        print(f"Received message: {msg}")
        
        # Determine if this is a medical query or casual conversation
        classification_response = llm.invoke(classifier_prompt.format(input=msg))
        message_type = classification_response.content.strip().upper()
        
        print(f"Message classified as: {message_type}")
        
        if message_type == "CASUAL":
            # Handle casual conversation directly with the LLM
            response = llm.invoke(conversation_prompt + "\n\nUser message: " + msg)
            answer = response.content
        else:
            # Create a context-aware query by including recent conversation history
            enhanced_query = ""
            
            # Only include previous context if there is more than one message
            if len(conversation_context) > 1:
                enhanced_query += "Previous conversation:\n"
                # Include the last few exchanges for context
                for i in range(len(conversation_context) - 1):
                    entry = conversation_context[i]
                    role = "User" if entry["role"] == "user" else "Assistant"
                    enhanced_query += f"{role}: {entry['content']}\n"
                
                enhanced_query += "\nCurrent question: " + msg
            else:
                enhanced_query = msg
            
            print(f"Enhanced query with context: {enhanced_query}")
            
            # Get response from RAG system
            response = rag_chain.invoke({"input": enhanced_query})
            answer = response.get("answer", "I don't have enough information to answer that question comprehensively.")
        
        # Add bot response to conversation history
        session['conversation_history'].append({"role": "assistant", "content": answer})
        session.modified = True
        
        print(f"Response: {answer}")
        return answer
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"There was an error processing your request. Please try asking again in a different way.", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)