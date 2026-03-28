# 🩺MediBot- Medical Chatbot
**RAG based Clinical Reference Assistant**

A Retrieval-Augmented Generation (RAG) powered medical chatbot that provides context aware, source grounded answers from a curated medical knowledge base(in this case : **The 3rd edition of The Gale Encyclopedia of Medicine**).

Instead of relying purely on LLM memory (which can hallucinate), this system retrieves relevant information from trusted documents using FAISS vector search, ensuring accurate and explainable responses.

<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/92a5931c-c58f-4aa3-b65e-1eafbad49c56" />
<img width="2560" height="1664" alt="image" src="https://github.com/user-attachments/assets/267ab058-67cf-49c8-83e3-4c33d7576112" />
<img width="2560" height="1664" alt="image" src="https://github.com/user-attachments/assets/db1644f1-a8f0-434d-8b82-929e7a972546" />

# ✨ Key Features
FAISS vector store for fast semantic retrieval

SentenceTransformer embeddings (all-MiniLM-L6-v2) 

Modular prompt template injection

Groq LLM backends

Source document traceability (shows which chunks supported the answer)

Caching of vector store + embeddings via Streamlit resource cache

# ⚙️ Tech Stack
Frontend: Streamlit
Backend: Python + LangChain
LLM Layer: Groq (LLaMA Model via Groq API)
Embedding Model: HuggingFace SentenceTransformers (all-MiniLM-L6-v2)
Vector DB: FAISS
Data Source: Medical PDFs (via PyPDFLoader)

# 🏗 Architecture

PDF(s) --> Text Splitter --> Embeddings(Hugging Face) --> FAISS Index (vectorstore/db_faiss)
								│
User Query --> Retriever (top-k) ---------------┘
			    │
		    Prompt Assembly
			    │
		    LLM Generation (Groq)
			    │
		    Answer + Source Chunks

## 📂 Main Components

| File/Directory               | Role                                                                 |
|-----------------------------|----------------------------------------------------------------------|
| `create_memory_for_llm.py`  | Builds FAISS index from PDFs (embedding + persistence)               |
| `connect_memory_with_llm.py`| CLI-based RAG querying using HuggingFace Endpoint                    |
| `medibot.py`                | Streamlit chatbot UI using Groq LLM + FAISS retrieval                |
| `vectorstore/db_faiss`      | Persisted FAISS vector store (precomputed embeddings)                |
| `data/`                     | Source medical PDF documents                                         |
