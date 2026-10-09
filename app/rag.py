"""RAG service: one FAISS index per uploaded document.

Answers come with the sources the model says it used. When the document does not contain the
answer, the model says so in the language of the question and no sources are returned.
"""

from __future__ import annotations

import re
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

# Used when the model refuses without giving an explanation.
NOT_FOUND_ANSWER = "I could not find this in the document."

PROMPT = """You answer questions about a single PDF document.
Use ONLY the numbered passages below. Do not use outside knowledge.
Answer in the same language as the question.

If the passages do not contain the answer, write the single word NOT_FOUND on the first line, then
one short sentence, in the language of the question, saying that the document does not contain
this information. Add nothing else.

Otherwise, answer the question. Then, on a new final line, write USED: followed by the numbers of
the passages you actually relied on, separated by commas (for example: USED: 1,3).

Passages:
{context}

Question: {question}"""


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
    found: bool
    sources: list[Source]


@dataclass(frozen=True)
class ParsedOutput:
    found: bool
    text: str
    used: list[int] | None  # 1-based passage numbers; None = the model did not say


@dataclass
class _Indexed:
    info: DocumentInfo
    store: FAISS


_NOT_FOUND_RE = re.compile(r"^\W*NOT_FOUND\b[\s:.\-–—*]*", re.IGNORECASE)
_USED_RE = re.compile(r"(?:^|\n)[ \t*]*USED[ \t*]*:[ \t]*([^\n]*?)[ \t*]*\s*$", re.IGNORECASE)


def parse_model_output(raw: str, n_passages: int) -> ParsedOutput:
    """Split the model's raw text into: found or not, the answer, and the passages it used."""
    text = raw.strip()
    marker = _NOT_FOUND_RE.match(text)
    if marker or text.lower().rstrip(".") == NOT_FOUND_ANSWER.lower().rstrip("."):
        explanation = text[marker.end() :].strip() if marker else ""
        return ParsedOutput(found=False, text=explanation or NOT_FOUND_ANSWER, used=[])

    used = None
    tail = _USED_RE.search(text)
    if tail:
        text = text[: tail.start()].rstrip()
        numbers = {int(n) for n in re.findall(r"\d+", tail.group(1))}
        used = sorted(n for n in numbers if 1 <= n <= n_passages) or None
    if not text:
        return ParsedOutput(found=False, text=NOT_FOUND_ANSWER, used=[])
    return ParsedOutput(found=True, text=text, used=used)


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

    def retrieve(self, doc_id: str, question: str) -> list[Document]:
        """The `top_k` chunks of this document closest to the question (no LLM call)."""
        with self._lock:
            item = self._docs.get(doc_id)
            if item is None:
                raise DocumentNotFound(doc_id)
            self._docs.move_to_end(doc_id)
        return item.store.similarity_search(question, k=self._settings.top_k)

    def ask(self, doc_id: str, question: str) -> Answer:
        hits = self.retrieve(doc_id, question)
        context = "\n\n".join(
            f"[{i}] (page {page_of(d)}) {d.page_content}" for i, d in enumerate(hits, start=1)
        )
        raw = self._chain.invoke({"context": context, "question": question})
        parsed = parse_model_output(raw, len(hits))

        if not parsed.found:
            chosen: list[Document] = []
        elif parsed.used is None:
            chosen = hits  # the model did not say which passages it used: show them all
        else:
            chosen = [hits[n - 1] for n in parsed.used]
        sources = [Source(page=page_of(d), excerpt=d.page_content[:200]) for d in chosen]
        return Answer(answer=parsed.text, found=parsed.found, sources=sources)

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            return [item.info for item in self._docs.values()]

    def delete(self, doc_id: str) -> None:
        with self._lock:
            if self._docs.pop(doc_id, None) is None:
                raise DocumentNotFound(doc_id)


def page_of(doc: Document) -> int:
    """1-based page number of a chunk."""
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
