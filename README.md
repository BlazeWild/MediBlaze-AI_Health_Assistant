# MediBlazeAI - AI Health Assistant

## Docker Deployment

### Prerequisites

- Docker and Docker Compose installed on your machine
- API keys for:
  - Pinecone (vector database)
  - Google Gemini (AI model)

### Environment Variables

Create a `.env` file with the following variables:

```
PINECONE_API_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_gemini_api_key
FLASK_SECRET_KEY=your_secret_key_for_flask
```

### Deploy with Docker Compose

1. Build and start the application:

   ```
   docker-compose up -d
   ```

2. The application will be available at: http://localhost:8000

3. To stop the application:
   ```
   docker-compose down
   ```

### Manual Docker Deployment

1. Build the Docker image:

   ```
   docker build -t medical-chatbot .
   ```

2. Run the container:

   ```
   docker run -p 8000:8000 \
     -e PINECONE_API_KEY=your_pinecone_api_key \
     -e GEMINI_API_KEY=your_gemini_api_key \
     -e FLASK_SECRET_KEY=your_secret_key \
     -d medical-chatbot
   ```

3. Access the application at: http://localhost:8000
