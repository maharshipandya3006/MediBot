import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_FAISS_PATH = Path("vectorstore/db_faiss")


def get_groq_llm():
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_model_name = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-20b")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file or Vercel environment variables.")

    return ChatGroq(
        model=groq_model_name,
        temperature=0.5,
        max_tokens=512,
        api_key=groq_api_key,
    )


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_vectorstore():
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(
            f"FAISS database not found at {DB_FAISS_PATH}. Run create_memory_for_llm.py first."
        )

    embedding_model = get_embedding_model()

    return FAISS.load_local(
        str(DB_FAISS_PATH),
        embedding_model,
        allow_dangerous_deserialization=True,
    )


def create_rag_chain():
    llm = get_groq_llm()
    db = load_vectorstore()

    prompt = PromptTemplate(
        template="""
You are MediBot, a helpful medical assistant.
Use only the given context to answer the user's question.
If the answer is not available in the context, say that you do not know.
Do not make up medical facts.
For serious symptoms, advise the user to consult a qualified medical professional.

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
        db.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain,
    )

    return rag_chain


def main():
    rag_chain = create_rag_chain()

    user_query = input("Write Query Here: ")

    if not user_query.strip():
        print("Please enter a valid question.")
        return

    response = rag_chain.invoke({"input": user_query})

    print("\nRESULT:")
    print(response["answer"])

    print("\nSOURCE DOCUMENTS:")
    for doc in response.get("context", []):
        source = doc.metadata.get("source", "Unknown source")
        page = doc.metadata.get("page", "Unknown page")
        preview = doc.page_content[:200].replace("\n", " ")

        print(f"- Source: {source}, Page: {page}")
        print(f"  Preview: {preview}...")


if __name__ == "__main__":
    main()