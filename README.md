# MediBot

MediBot is a full-stack medical reference assistant built with FastAPI, LangChain,
Groq, HuggingFace embeddings, and FAISS. The deployed Render URL opens a clean
chat interface, while the API remains available for integrations.

The assistant answers from a local medical reference index and returns source
snippets so users can see which retrieved context supported the response.

## What Is Included

- Professional responsive chat UI served by FastAPI
- `POST /api/ask` endpoint for chatbot questions
- Legacy `POST /ask` endpoint for existing clients
- `GET /health` endpoint for uptime checks
- Cached embedding model, vector store, LLM client, and RAG chain
- Source snippets in every successful chatbot response
- Medical safety-oriented system prompt
- FAISS index generated from `data/` PDFs

## Project Structure

```text
.
|-- app.py                       # FastAPI app, static UI, API routes, RAG chain
|-- create_memory_for_llm.py      # Builds vectorstore/db_faiss from PDFs
|-- connect_memory_with_llm.py    # CLI querying helper
|-- static/
|   |-- index.html                # Chat UI
|   |-- styles.css                # Responsive visual system
|   |-- app.js                    # Frontend API integration
|   `-- favicon.svg               # Browser tab icon
|-- data/                         # Source medical PDF files
|-- vectorstore/db_faiss/         # Persisted FAISS index
|-- requirements.txt
|-- render.yaml
`-- .env.example
```

## Local Setup

Use Python 3.13.2.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your Groq key in `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=openai/gpt-oss-20b
```

Start the app:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Rebuild The Vector Store

If you change PDFs inside `data/`, rebuild the FAISS index:

```bash
python create_memory_for_llm.py
```

This creates or updates:

```text
vectorstore/db_faiss/index.faiss
vectorstore/db_faiss/index.pkl
```

Only load FAISS stores that you trust. LangChain FAISS persistence uses pickle
metadata and the app must enable deserialization to restore the docstore.

## API

Health check:

```http
GET /health
```

Ask a question:

```http
POST /api/ask
Content-Type: application/json

{
  "question": "What are common symptoms of asthma?"
}
```

Response:

```json
{
  "question": "What are common symptoms of asthma?",
  "answer": "A source-grounded answer...",
  "sources": [
    {
      "source": "The Gale Encyclopedia of Medicine",
      "page": "123",
      "preview": "Retrieved source snippet..."
    }
  ]
}
```

Interactive API docs:

```text
/docs
```

## Render Deployment

Create a Render Web Service from this repository.

Recommended settings:

```text
Runtime: Python
Build Command: pip install -r requirements.txt
Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=openai/gpt-oss-20b
```

After deployment, your Render root URL opens the MediBot chat UI. The raw API is
still available at `/api/ask`, `/ask`, `/health`, and `/docs`.

## Medical Safety Note

MediBot is an educational reference assistant, not a diagnostic system. Users
should consult qualified healthcare professionals for personal medical decisions
and seek emergency help for urgent symptoms.
