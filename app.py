from flask import Flask, render_template, jsonify, request, session
from src.helpers import download_hugging_face_embeddings
from dotenv import load_dotenv
from src.prompt import *
import os
from datetime import datetime
from pinecone import Pinecone

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))  # For session management

# Load environment variables from .env file if it exists
load_dotenv()

# Get environment variables with error handling
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"PINECONE_API_KEY set: {'Yes' if PINECONE_API_KEY else 'No'}")
print(f"GEMINI_API_KEY set: {'Yes' if GEMINI_API_KEY else 'No'}")

# Use dummy data if environment variables are not set
use_dummy_data = False

if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY environment variable is not set. Using dummy mode.")
    use_dummy_data = True
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable is not set. Using dummy mode.")
    use_dummy_data = True

# Initialize embeddings
embeddings = download_hugging_face_embeddings()

# Set up a dummy index if needed
class DummyIndex:
    def query(self, **kwargs):
        class DummyResponse:
            def __init__(self):
                class DummyMatch:
                    def __init__(self):
                        self.metadata = {"text": "Acne is a common skin condition characterized by whiteheads, blackheads, pimples, and deeper lumps like cysts. It typically occurs when hair follicles get clogged with oil and dead skin cells. Acne most commonly appears on the face, neck, chest, back, and shoulders."}
                self.matches = [DummyMatch()]
        return DummyResponse()

# Initialize Pinecone if possible
if not use_dummy_data:
    try:
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Get the Pinecone index
        index_name = "medical-chatbot"
        try:
            # Check if index exists
            indexes = pc.list_indexes()
            print(f"Available indexes: {indexes}")
            
            if not index_name in [idx.name for idx in indexes]:
                print(f"Warning: Index '{index_name}' not found!")
                use_dummy_data = True
            else:
                index = pc.Index(index_name)
                # Test query to verify index is working
                test_embedding = embeddings.embed_query("test")
                test_result = index.query(vector=test_embedding, top_k=1, include_metadata=True)
                print(f"Test query successful: {test_result}")
        except Exception as e:
            print(f"Error initializing Pinecone index: {str(e)}")
            use_dummy_data = True
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        use_dummy_data = True

# Use dummy index if needed
if use_dummy_data:
    index = DummyIndex()
    print("Using dummy index for fallback functionality")

# Function to get similar documents from Pinecone
def get_similar_docs(query, k=10):
    try:
        # Convert query to embedding vector
        query_embedding = embeddings.embed_query(query)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Extract documents from results
        docs = []
        if hasattr(results, 'matches'):
            docs = [item.metadata.get("text", "") for item in results.matches if hasattr(item, 'metadata')]
        else:
            # Alternative extraction if the API structure is different
            print("Using alternative result extraction method")
            if isinstance(results, dict) and 'matches' in results:
                docs = [match.get('metadata', {}).get('text', '') for match in results['matches']]
        
        # If we couldn't get any docs, return a default response
        if not docs:
            docs = ["Acne is a common skin condition characterized by whiteheads, blackheads, pimples, and deeper lumps like cysts. It typically occurs when hair follicles get clogged with oil and dead skin cells. Acne most commonly appears on the face, neck, chest, back, and shoulders."]
        
        return docs
    except Exception as e:
        print(f"Error in get_similar_docs: {str(e)}")
        # Return a fallback document set when something goes wrong
        return ["Acne is a common skin condition characterized by whiteheads, blackheads, pimples, and deeper lumps like cysts. It typically occurs when hair follicles get clogged with oil and dead skin cells. Acne most commonly appears on the face, neck, chest, back, and shoulders."]

# Initialize Google's Gemini model if available
try:
    if not use_dummy_data and GEMINI_API_KEY:
        from google.generativeai import GenerativeModel, configure
        configure(api_key=GEMINI_API_KEY)
        model = GenerativeModel("gemini-1.5-flash")
        print("Google Gemini model initialized")
    else:
        # Create a dummy model class for testing
        class DummyModel:
            def generate_content(self, prompt):
                class DummyResponse:
                    def __init__(self):
                        if "CASUAL" in prompt:
                            self.text = "CASUAL"
                        elif "determine if this message" in prompt.lower():
                            self.text = "MEDICAL" if any(word in prompt.lower() for word in ["illness", "disease", "acne", "treatment", "doctor", "medicine", "medical"]) else "CASUAL"
                        else:
                            self.text = "I'm a dummy response. In a real environment, I would provide helpful medical information based on your question."
                return DummyResponse()
        model = DummyModel()
        print("Using dummy model for testing")
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
    # Create a dummy model class for testing
    class DummyModel:
        def generate_content(self, prompt):
            class DummyResponse:
                def __init__(self):
                    if "CASUAL" in prompt:
                        self.text = "CASUAL"
                    elif "determine if this message" in prompt.lower():
                        self.text = "MEDICAL" if any(word in prompt.lower() for word in ["illness", "disease", "acne", "treatment", "doctor", "medicine", "medical"]) else "CASUAL"
                    else:
                        self.text = "I'm a dummy response. In a real environment, I would provide helpful medical information based on your question."
            return DummyResponse()
    model = DummyModel()
    print("Using dummy model due to initialization error")

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
        classification_prompt = """Determine if this message is:
         1. A casual greeting or non-medical conversation (CASUAL)
         2. A message that requires medical knowledge to answer (MEDICAL)
         Respond with only one word: either CASUAL or MEDICAL.
         
         Message: {0}
        """
        
        classification_response = model.generate_content(classification_prompt.format(msg))
        message_type = classification_response.text.strip().upper()
        
        print(f"Message classified as: {message_type}")
        
        if message_type == "CASUAL":
            # Handle casual conversation directly with the LLM
            conversation_prompt = """You're a friendly health assistant. Respond to this casual conversation in a helpful, friendly way.
            Keep your response relatively short and conversational.
            
            User message: {0}
            """
            
            response = model.generate_content(conversation_prompt.format(msg))
            answer = response.text
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
            
            # Get similar documents from Pinecone
            context_docs = get_similar_docs(enhanced_query)
            context_text = "\n\n".join(context_docs)
            
            # Prepare RAG prompt
            rag_prompt = f"""{system_prompt}
            
            Context information:
            {context_text}
            
            User question: {enhanced_query}
            """
            
            # Generate response
            response = model.generate_content(rag_prompt)
            answer = response.text
        
        # Add bot response to conversation history
        session['conversation_history'].append({"role": "assistant", "content": answer})
        session.modified = True
        
        print(f"Response: {answer}")
        return answer
    except Exception as e:
        print(f"Error: {str(e)}")
        return f"There was an error processing your request. Please try asking again in a different way.", 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Get port from environment variable (required for Railway)
    port = int(os.environ.get('PORT', 8080))
    # Make sure to bind to 0.0.0.0 for Railway
    app.run(host="0.0.0.0", port=port, debug=False)