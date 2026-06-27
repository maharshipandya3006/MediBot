import os

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = FastAPI()

DB_FAISS_PATH = "vectorstore/db_faiss"


class QuestionRequest(BaseModel):
    question: str


def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


def get_rag_chain():
    vectorstore = get_vectorstore()

    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_model_name = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-20b")

    llm = ChatGroq(
        model=groq_model_name,
        temperature=0.5,
        max_tokens=512,
        api_key=groq_api_key,
    )

    prompt = PromptTemplate(
        template="""
Use the given context to answer the user's medical question.
If you do not know the answer, say that you do not know.
Do not make up medical information.

Context:
{context}

Question:
{input}

Answer:
""",
        input_variables=["context", "input"],
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 6}),
        combine_docs_chain
    )

    return rag_chain


@app.get("/")
def home():
    return {
        "message": "MediBot API is running. Use POST /ask to ask a question."
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    rag_chain = get_rag_chain()
    response = rag_chain.invoke({"input": request.question})

    return {
        "question": request.question,
        "answer": response["answer"]
    }