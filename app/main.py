"""FastAPI application: upload PDFs, then ask questions about a specific document."""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
from functools import lru_cache

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.rag import DocumentNotFound, InvalidDocument, RagService, build_service

logger = logging.getLogger("rag")

app = FastAPI(title="RAG Document QA API", version="0.2.0")

_settings = get_settings()
if _settings.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_settings.allowed_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )


# ---------- schemas ----------
class QuestionIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class SourceOut(BaseModel):
    page: int
    excerpt: str


class AnswerOut(BaseModel):
    answer: str
    sources: list[SourceOut]


class DocumentOut(BaseModel):
    id: str
    filename: str
    pages: int
    chunks: int


# ---------- dependencies ----------
def require_api_key(
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.api_key and not secrets.compare_digest(x_api_key or "", settings.api_key):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


@lru_cache(maxsize=1)
def _build(settings: Settings) -> RagService:
    return build_service(settings)


def get_service(settings: Settings = Depends(get_settings)) -> RagService:
    try:
        return _build(settings)
    except RuntimeError as exc:  # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ---------- routes ----------
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentOut, status_code=201)
def upload_document(
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
    settings: Settings = Depends(get_settings),
):
    # The client-supplied name is only kept for display; it is never used as a path.
    filename = os.path.basename(file.filename or "document.pdf")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", dir=settings.tmp_dir, delete=False) as out:
            tmp_path = out.name
            head = file.file.read(5)
            if head != b"%PDF-":
                raise HTTPException(status_code=400, detail="The uploaded file is not a PDF.")
            out.write(head)
            size = len(head)
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {settings.max_upload_mb} MB).",
                    )
                out.write(chunk)
        info = service.ingest(tmp_path, filename)
    except InvalidDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
    return DocumentOut(id=info.id, filename=info.filename, pages=info.pages, chunks=info.chunks)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(_: None = Depends(require_api_key), service: RagService = Depends(get_service)):
    return [DocumentOut(**vars(d)) for d in service.list_documents()]


@app.post("/documents/{doc_id}/ask", response_model=AnswerOut)
def ask(
    doc_id: str,
    body: QuestionIn,
    _: None = Depends(require_api_key),
    service: RagService = Depends(get_service),
):
    try:
        result = service.ask(doc_id, body.question)
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail="Unknown document id.") from exc
    except Exception as exc:
        logger.exception("LLM or retrieval failure")
        raise HTTPException(status_code=502, detail="The language model request failed.") from exc
    return AnswerOut(
        answer=result.answer,
        sources=[SourceOut(page=s.page, excerpt=s.excerpt) for s in result.sources],
    )


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(
    doc_id: str, _: None = Depends(require_api_key), service: RagService = Depends(get_service)
):
    try:
        service.delete(doc_id)
    except DocumentNotFound as exc:
        raise HTTPException(status_code=404, detail="Unknown document id.") from exc
