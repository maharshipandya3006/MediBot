import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = FastAPI()

DB_FAISS_PATH = Path("vectorstore/db_faiss")


class QuestionRequest(BaseModel):
    question: str


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_vectorstore():
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError("FAISS database not found at vectorstore/db_faiss.")

    embedding_model = get_embedding_model()

    return FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def create_rag_chain():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_model_name = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-20b")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    llm = ChatGroq(
        model=groq_model_name,
        temperature=0.5,
        max_tokens=512,
        api_key=groq_api_key,
    )

    vectorstore = load_vectorstore()

    prompt = PromptTemplate(
        template="""
You are MediBot,a helpful medical assistant.
Use only the provided context to answer the user's question.
If the answer is not available in the context,say that you do not know.
Do not make up medical facts.
For serious symptoms,advise the user to consult a qualified medical professional.

Context:
{context}

Question:
{input}

Answer:
""",
        input_variables=["context", "input"],
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain,
    )


@app.get("/")
def home():
    return {
        "message": "MediBot API is running. Use POST /ask to ask a question."
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        rag_chain = create_rag_chain()
        response = rag_chain.invoke({"input": question})

        return {
            "question": question,
            "answer": response["answer"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))