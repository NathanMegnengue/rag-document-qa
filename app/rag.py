"""RAG service: one FAISS index per uploaded document, answers come with their sources."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import Settings

NOT_FOUND_ANSWER = "I could not find this in the document."

PROMPT = f"""You answer questions about a single PDF document.
Use ONLY the context below. If the context does not contain the answer, reply exactly:
"{NOT_FOUND_ANSWER}"
Do not use outside knowledge.

Context:
{{context}}

Question: {{question}}"""


class DocumentNotFound(Exception):
    """No document with this id (never uploaded, deleted, or evicted)."""


class InvalidDocument(Exception):
    """The PDF is unreadable or contains no extractable text."""


@dataclass(frozen=True)
class Source:
    page: int  # 1-based
    excerpt: str


@dataclass(frozen=True)
class DocumentInfo:
    id: str
    filename: str
    pages: int
    chunks: int


@dataclass(frozen=True)
class Answer:
    answer: str
    sources: list[Source]


@dataclass
class _Indexed:
    info: DocumentInfo
    store: FAISS


class RagService:
    """Holds the indexes in memory. `embeddings` and `llm` are injected, so tests can fake them."""

    def __init__(self, embeddings, llm, settings: Settings):
        self._embeddings = embeddings
        self._settings = settings
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self._chain = ChatPromptTemplate.from_template(PROMPT) | llm | StrOutputParser()
        self._docs: OrderedDict[str, _Indexed] = OrderedDict()
        self._lock = threading.Lock()

    def ingest(self, pdf_path: str, filename: str) -> DocumentInfo:
        try:
            reader = PdfReader(pdf_path)
            pages = [
                Document(page_content=page.extract_text() or "", metadata={"page": i})
                for i, page in enumerate(reader.pages)
            ]
        except Exception as exc:  # pypdf raises many different error types on bad files
            raise InvalidDocument("This file could not be read as a PDF.") from exc

        chunks = [c for c in self._splitter.split_documents(pages) if c.page_content.strip()]
        if not chunks:
            raise InvalidDocument("This PDF has no extractable text (scanned document?).")

        store = FAISS.from_documents(chunks, self._embeddings)
        info = DocumentInfo(
            id=uuid.uuid4().hex, filename=filename, pages=len(pages), chunks=len(chunks)
        )
        with self._lock:
            self._docs[info.id] = _Indexed(info, store)
            while len(self._docs) > self._settings.max_documents:
                self._docs.popitem(last=False)  # evict the oldest document
        return info

    def ask(self, doc_id: str, question: str) -> Answer:
        with self._lock:
            item = self._docs.get(doc_id)
            if item is None:
                raise DocumentNotFound(doc_id)
            self._docs.move_to_end(doc_id)

        hits = item.store.similarity_search(question, k=self._settings.top_k)
        context = "\n\n".join(f"[page {_page(d)}] {d.page_content}" for d in hits)
        answer = self._chain.invoke({"context": context, "question": question})
        sources = [Source(page=_page(d), excerpt=d.page_content[:200]) for d in hits]
        return Answer(answer=answer.strip(), sources=sources)

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            return [item.info for item in self._docs.values()]

    def delete(self, doc_id: str) -> None:
        with self._lock:
            if self._docs.pop(doc_id, None) is None:
                raise DocumentNotFound(doc_id)


def _page(doc) -> int:
    return int(doc.metadata.get("page", 0)) + 1


def build_service(settings: Settings) -> RagService:
    """Create the real service. Heavy imports are done here, not at module import time."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return RagService(embeddings, llm, settings)
