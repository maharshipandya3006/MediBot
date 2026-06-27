import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = Path("data")
DB_FAISS_PATH = Path("vectorstore/db_faiss")


def load_pdf_files(data_path: Path):
    if not data_path.exists():
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    loader = DirectoryLoader(
        str(data_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        raise ValueError("No PDF files found inside the data folder.")

    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    text_chunks = text_splitter.split_documents(documents)

    if not text_chunks:
        raise ValueError("No text chunks were created from the PDFs.")

    return text_chunks


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_faiss_database():
    print("Loading PDF files...")
    documents = load_pdf_files(DATA_PATH)
    print("Length of PDF pages:", len(documents))

    print("Creating text chunks...")
    text_chunks = create_chunks(documents)
    print("Length of text chunks:", len(text_chunks))

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating FAISS vector database...")
    DB_FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)

    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(str(DB_FAISS_PATH))

    print(f"FAISS vector database saved successfully at: {DB_FAISS_PATH}")


if __name__ == "__main__":
    create_faiss_database()