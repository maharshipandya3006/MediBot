import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field
from groq import AuthenticationError

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = Path(__file__).resolve().parent
DB_FAISS_PATH = BASE_DIR / "vectorstore" / "db_faiss"
STATIC_DIR = BASE_DIR / "static"

DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVER_K = 3
CASUAL_RESPONSES = {
    "hi": "Hello! How can I help you today?",
    "hii": "Hello! How can I help you today?",
    "hello": "Hello! How can I help you today?",
    "hey": "Hello! How can I help you today?",
    "good morning": "Good morning! How can I help you today?",
    "good afternoon": "Good afternoon! How can I help you today?",
    "good evening": "Good evening! How can I help you today?",
    "thanks": "You're welcome.",
    "thank you": "You're welcome.",
    "thank you so much": "You're welcome.",
    "who are you": (
        "I am MediBot, a medical reference assistant. I can help explain "
        "medical terms, symptoms, and conditions in simple language."
    ),
    "how are you": "I am ready to help. What would you like to ask?",
}
TOPIC_KEYWORDS = (
    "i have",
    "i am having",
    "i'm having",
    "having",
    "suffering from",
    "diagnosed with",
    "fever",
    "infection",
    "pain",
    "cough",
    "cold",
    "headache",
    "migraine",
    "asthma",
    "diabetes",
    "hypertension",
    "blood pressure",
    "bp",
    "throat",
    "stomach",
    "rash",
)
FOLLOW_UP_PATTERNS = (
    "it",
    "this",
    "that",
    "its",
    "for it",
    "about it",
    "treatment",
    "treat",
    "medicine",
    "medication",
    "symptoms",
    "causes",
    "how long",
    "what should i do",
    "what to do",
    "help me",
    "explain more",
)
MEDICAL_TOPIC_KEYWORDS = (
    "abdomen",
    "acne",
    "allergy",
    "antibiotic",
    "anxiety",
    "asthma",
    "back pain",
    "bacteria",
    "bleeding",
    "blood",
    "blood pressure",
    "bp",
    "breathing",
    "cancer",
    "chest",
    "cold",
    "cough",
    "diabetes",
    "diarrhea",
    "dizziness",
    "ear",
    "fever",
    "flu",
    "headache",
    "heart",
    "hypertension",
    "infection",
    "injury",
    "kidney",
    "liver",
    "migraine",
    "nausea",
    "pain",
    "pneumonia",
    "rash",
    "sinus",
    "skin",
    "sore",
    "stomach",
    "stress",
    "stroke",
    "swelling",
    "symptom",
    "throat",
    "tonsil",
    "urine",
    "virus",
    "vomit",
    "wound",
)
QUERY_EXPANSIONS = {
    "bp": "blood pressure hypertension high blood pressure",
    "blood pressure": "blood pressure hypertension high blood pressure",
    "high bp": "high blood pressure hypertension",
    "low bp": "low blood pressure hypotension",
    "sugar": "blood sugar glucose diabetes",
    "heart attack": "myocardial infarction heart attack",
    "stroke": "cerebrovascular accident stroke",
    "stomach pain": "abdominal pain stomach pain",
    "throat infection": "sore throat tonsillitis pharyngitis throat infection",
}

SYSTEM_PROMPT = """
You are MediBot, a careful medical reference assistant.
Use only the provided context to answer the user's question.
If the answer is not available in the context, say that you do not know.
Do not make up medical facts, diagnoses, or treatment plans.
Write in a clean, professional, human tone.
Do not use markdown formatting, asterisks, bold text, headings, citation labels,
source names, page numbers, or references to the encyclopedia.
Do not mention "context", "provided information", or "sources" in the answer.
Use short paragraphs. Use simple numbered points only when that improves clarity.
For urgent symptoms such as chest pain, severe breathing trouble, stroke symptoms,
severe allergic reaction, heavy bleeding, fainting, or suicidal thoughts, tell the
user to seek emergency medical help immediately.
Always remind the user that this is educational information and not a substitute
for care from a qualified medical professional.

Context:
{context}

Recent conversation:
{conversation}

Current question:
{input}

Answer:
"""


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=2000)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1200)
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)


class Source(BaseModel):
    source: str
    page: str
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    vectorstore_ready: bool
    model: str


app = FastAPI(
    title="MediBot",
    description="A source-grounded medical reference assistant powered by RAG.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def get_cors_origins() -> list[str]:
    origins = os.environ.get("CORS_ORIGINS", "*")
    parsed_origins = [origin.strip() for origin in origins.split(",") if origin.strip()]
    return parsed_origins or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def load_vectorstore() -> FAISS:
    if not DB_FAISS_PATH.exists():
        raise FileNotFoundError(
            "FAISS database not found at vectorstore/db_faiss. "
            "Run create_memory_for_llm.py before starting the app."
        )

    return FAISS.load_local(
        str(DB_FAISS_PATH),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_model_name = os.environ.get("GROQ_MODEL_NAME", DEFAULT_MODEL_NAME)

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is missing.")

    return ChatGroq(
        model=groq_model_name,
        temperature=0.2,
        max_tokens=700,
        api_key=groq_api_key,
    )


@lru_cache(maxsize=1)
def create_rag_chain() -> Any:
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "conversation", "input"],
    )
    combine_docs_chain = create_stuff_documents_chain(get_llm(), prompt)

    return create_retrieval_chain(
        load_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K}),
        combine_docs_chain,
    )


def build_sources(context: list[Any]) -> list[Source]:
    sources: list[Source] = []

    for doc in context:
        metadata = doc.metadata or {}
        source_name = format_source_name(metadata.get("source"))
        page = format_page_label(metadata)
        preview = " ".join(doc.page_content.split())[:260]

        sources.append(
            Source(
                source=source_name,
                page=page,
                preview=preview,
            )
        )

    return sources


def format_source_name(raw_source: Any) -> str:
    if not raw_source:
        return "Medical reference"

    filename = Path(str(raw_source)).name

    if "Gale-Encyclopedia-of-Medicine" in filename:
        return "The Gale Encyclopedia of Medicine"

    return filename or "Medical reference"


def format_page_label(metadata: dict[str, Any]) -> str:
    page_label = metadata.get("page_label")

    if page_label:
        return str(page_label)

    page = metadata.get("page")

    if isinstance(page, int):
        return str(page + 1)

    if isinstance(page, str) and page.isdigit():
        return str(int(page) + 1)

    return "Unknown"


def normalize_casual_question(question: str) -> str:
    normalized = question.lower().strip()
    return normalized.strip(".,!?;: ")


def get_casual_response(question: str) -> str | None:
    normalized = normalize_casual_question(question)

    if normalized in CASUAL_RESPONSES:
        return CASUAL_RESPONSES[normalized]

    if normalized in {"yo", "sup", "hello there", "hey there"}:
        return "Hello! How can I help you today?"

    return None


def build_conversation_context(history: list[ChatMessage]) -> str:
    scoped_history = scope_history_to_latest_topic(history)

    if not scoped_history:
        return "No previous conversation."

    lines: list[str] = []

    for message in scoped_history[-6:]:
        content = " ".join(message.content.split())

        if not content:
            continue

        label = "User" if message.role == "user" else "MediBot"
        lines.append(f"{label}: {content[:700]}")

    return "\n".join(lines) if lines else "No previous conversation."


def build_retrieval_input(question: str, history: list[ChatMessage]) -> str:
    expanded_question = expand_medical_query(question)

    if is_standalone_medical_question(question):
        return expanded_question

    if not is_likely_follow_up(question):
        return expanded_question

    conversation = build_conversation_context(history)

    if conversation == "No previous conversation.":
        return expanded_question

    return (
        "Use this previous conversation only to understand follow-up wording.\n"
        f"{conversation}\n\nCurrent question: {expanded_question}"
    )


def is_likely_topic_start(text: str) -> bool:
    normalized = text.lower().strip()

    if len(normalized) < 5:
        return False

    return is_standalone_medical_question(normalized)


def is_standalone_medical_question(text: str) -> bool:
    normalized = text.lower().strip()

    if len(normalized) < 3:
        return False

    return contains_medical_topic(normalized)


def is_likely_follow_up(text: str) -> bool:
    normalized = text.lower().strip(" .?!,:;")

    if not normalized:
        return False

    if contains_medical_topic(normalized):
        return False

    if len(normalized.split()) <= 6:
        return any(pattern in normalized for pattern in FOLLOW_UP_PATTERNS)

    return False


def contains_medical_topic(text: str) -> bool:
    normalized = text.lower().strip()
    return any(keyword in normalized for keyword in MEDICAL_TOPIC_KEYWORDS)


def expand_medical_query(question: str) -> str:
    normalized = question.lower()
    expansions: list[str] = []

    for trigger, expansion in QUERY_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(trigger)}\b", normalized):
            expansions.append(expansion)

    if not expansions:
        return question

    return f"{question}\nRelated medical terms: {'; '.join(expansions)}"


def scope_history_to_latest_topic(history: list[ChatMessage]) -> list[ChatMessage]:
    if not history:
        return []

    start_index = 0

    for index in range(len(history) - 1, -1, -1):
        message = history[index]

        if message.role == "user" and is_likely_topic_start(message.content):
            start_index = index
            break

    return history[start_index:]


def clean_answer_text(answer: str) -> str:
    cleaned = answer.replace("**", "")
    cleaned = cleaned.replace("*", "")
    cleaned = cleaned.replace("__", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"(?m)^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(
        r"(?im)^ *(source|sources|references?|citation|citations):.*$",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?im)^.*gale encyclopedia.*$", "", cleaned)
    cleaned = re.sub(r"(?im)^.*\bpage +[ivxlcdm0-9]+\b.*$", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def answer_question(question: str, history: list[ChatMessage] | None = None) -> AskResponse:
    clean_question = question.strip()
    safe_history = history or []

    if not clean_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    casual_response = get_casual_response(clean_question)

    if casual_response:
        return AskResponse(
            question=clean_question,
            answer=casual_response,
            sources=[],
        )

    try:
        conversation = build_conversation_context(safe_history)

        if is_standalone_medical_question(clean_question) or not is_likely_follow_up(
            clean_question
        ):
            conversation = "No previous conversation."

        response = create_rag_chain().invoke(
            {
                "input": build_retrieval_input(clean_question, safe_history),
                "conversation": conversation,
            }
        )
    except AuthenticationError as exc:
        raise HTTPException(
            status_code=503,
            detail="Groq authentication failed. Check GROQ_API_KEY.",
        ) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="MediBot could not generate an answer right now. Please try again.",
        ) from exc

    return AskResponse(
        question=clean_question,
        answer=clean_answer_text(
            response.get("answer", "I do not know based on the available context.")
        ),
        sources=[],
    )


@app.get("/", include_in_schema=False, response_model=None)
def home():
    index_file = STATIC_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return {"message": "MediBot API is running. Use POST /api/ask to ask a question."}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vectorstore_ready=DB_FAISS_PATH.exists(),
        model=os.environ.get("GROQ_MODEL_NAME", DEFAULT_MODEL_NAME),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask_question(request: QuestionRequest) -> AskResponse:
    return answer_question(request.question, request.history)


@app.post("/ask", response_model=AskResponse)
def ask_question_legacy(request: QuestionRequest) -> AskResponse:
    return answer_question(request.question, request.history)
