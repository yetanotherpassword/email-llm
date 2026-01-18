"""FastAPI web application for email-llm."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .config import settings
from .embeddings import VectorStore
from .llm_client import LLMClient
from .models import AttachmentType, SearchQuery
from .rag_engine import RAGEngine
from .yolo_analyzer import get_available_object_classes

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Email-LLM",
    description="Natural language search for your emails",
    version="0.1.0",
)

# Static files and templates
static_dir = Path(__file__).parent.parent / "static"
templates_dir = Path(__file__).parent.parent / "templates"

static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize components (lazy loading)
_vector_store: Optional[VectorStore] = None
_llm_client: Optional[LLMClient] = None
_rag_engine: Optional[RAGEngine] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine(
            vector_store=get_vector_store(),
            llm_client=get_llm_client(),
        )
    return _rag_engine


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    limit: int = 20
    use_llm: bool = True
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    from_address: Optional[str] = None
    folder: Optional[str] = None
    has_attachments: Optional[bool] = None
    attachment_type: Optional[str] = None
    has_faces: Optional[bool] = None
    contains_object: Optional[str] = None


class AskRequest(BaseModel):
    question: str


class SearchResponse(BaseModel):
    query: str
    answer: Optional[str] = None
    results: list[dict]
    total_found: int


# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "object_classes": get_available_object_classes(),
        },
    )


@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search emails with natural language."""
    rag = get_rag_engine()

    # Build search query with filters
    from_date = None
    to_date = None

    if request.from_date:
        try:
            from_date = datetime.fromisoformat(request.from_date)
        except ValueError:
            pass

    if request.to_date:
        try:
            to_date = datetime.fromisoformat(request.to_date)
        except ValueError:
            pass

    attachment_type = None
    if request.attachment_type:
        try:
            attachment_type = AttachmentType(request.attachment_type)
        except ValueError:
            pass

    # Use simple search or filtered search
    if any([from_date, to_date, request.from_address, request.folder,
            request.has_attachments, attachment_type]):
        query = SearchQuery(
            query=request.query,
            limit=request.limit,
            from_date=from_date,
            to_date=to_date,
            from_address=request.from_address,
            folder=request.folder,
            has_attachments=request.has_attachments,
            attachment_type=attachment_type,
            has_faces=request.has_faces,
            contains_object=request.contains_object,
        )
        results = await rag.search_with_filters(query)
    else:
        results = await rag.search(
            request.query,
            limit=request.limit,
            use_llm=request.use_llm,
        )

    return SearchResponse(
        query=request.query,
        answer=results.get("answer"),
        results=results.get("results", []),
        total_found=results.get("total_found", 0),
    )


@app.post("/api/ask")
async def ask(request: AskRequest):
    """Ask a question and get a direct answer."""
    rag = get_rag_engine()
    answer = await rag.ask(request.question)
    return {"question": request.question, "answer": answer}


@app.get("/api/ask/stream")
async def ask_stream(question: str = Query(...)):
    """Stream the answer to a question."""
    rag = get_rag_engine()
    llm = get_llm_client()

    # Get relevant context first
    results = await rag.search(question, limit=5, use_llm=False)

    # Build context
    context_parts = []
    for i, result in enumerate(results.get("results", [])[:5], 1):
        context = f"Email {i}:\n"
        if result.get("subject"):
            context += f"Subject: {result['subject']}\n"
        if result.get("from_address"):
            context += f"From: {result['from_address']}\n"
        context += f"Content:\n{result['matched_text']}\n"
        context_parts.append(context)

    context = "\n---\n".join(context_parts)

    prompt = f"""User's question: {question}

Retrieved emails:
{context}

Based on the emails above, please answer the user's question."""

    async def generate():
        async for chunk in llm.generate_stream(
            prompt=prompt,
            system_prompt="You are an email search assistant. Answer questions based on the provided email context.",
            temperature=0.3,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


@app.get("/api/stats")
async def stats():
    """Get index statistics."""
    vs = get_vector_store()
    return {
        "total_chunks": vs.count(),
        "collection": vs.collection_name,
    }


@app.get("/api/objects")
async def objects():
    """Get list of detectable objects."""
    return {"objects": get_available_object_classes()}


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    llm = get_llm_client()
    llm_connected = await llm.check_connection()

    return {
        "status": "healthy",
        "llm_connected": llm_connected,
        "index_chunks": get_vector_store().count(),
    }
