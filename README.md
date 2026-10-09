# RAG Document QA

A small REST API to ask questions about PDF documents. You upload a PDF, get a document id, and
ask questions about that document. Every answer comes with the pages it was based on.

Built with FastAPI, LangChain, FAISS, `sentence-transformers` embeddings and LLaMA 3.3 70B
(through the Groq API).

> **Status: a tested prototype.** The API is isolated per document, validates uploads and has
> automated tests and CI. It is **not** a hardened production service: see
> [Limitations](#limitations).

## How it works

```text
POST /documents  (PDF)
   -> check it is a PDF and under the size limit
   -> extract text per page (pypdf)
   -> split into chunks (500 characters, 50 overlap)
   -> embed with all-MiniLM-L6-v2 -> one FAISS index per document

POST /documents/{id}/ask  {"question": "..."}
   -> retrieve the 3 closest chunks of that document
   -> prompt LLaMA 3.3 with the chunks, told to answer only from them
   -> return the answer + the pages and excerpts used
```

If the answer is not in the retrieved text, the model is instructed to reply
`I could not find this in the document.` This is a prompt instruction, not a guarantee: the
quality of the answers has not been measured yet (see the roadmap).

## Quick start

```bash
git clone https://github.com/NathanMegnengue/rag-document-qa.git
cd rag-document-qa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then put your GROQ_API_KEY in it
uvicorn app.main:app --reload
```

The first request that needs the models downloads the embedding model, so it is slow once.
Interactive docs: http://127.0.0.1:8000/docs

With Docker:

```bash
docker build -t rag-document-qa .
docker run --rm -p 8000:8000 --env-file .env rag-document-qa
```

## API

```bash
# 1. Upload a PDF -> returns an id
curl -X POST http://127.0.0.1:8000/documents -F "file=@report.pdf"
# {"id":"3f2a...","filename":"report.pdf","pages":12,"chunks":58}

# 2. Ask a question about that document
curl -X POST http://127.0.0.1:8000/documents/3f2a.../ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main conclusions?"}'
# {"answer":"...","sources":[{"page":4,"excerpt":"..."}, ...]}
```

| Endpoint | Purpose | Errors |
|---|---|---|
| `GET /health` | liveness check | |
| `POST /documents` | upload and index a PDF | 400 not a PDF, 413 too large, 422 unreadable or no text |
| `GET /documents` | list indexed documents | |
| `POST /documents/{id}/ask` | ask a question | 404 unknown id, 422 empty question, 502 LLM failure |
| `DELETE /documents/{id}` | remove a document | 404 unknown id |

## Configuration

Environment variables (or `.env`, see `.env.example`):

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none (required) | Groq API key |
| `API_KEY` | unset | if set, clients must send `X-API-Key` (except on `/health`) |
| `ALLOWED_ORIGINS` | empty | comma-separated CORS origins; empty means CORS disabled |
| `MAX_UPLOAD_MB` | 20 | upload size limit |
| `MAX_DOCUMENTS` | 20 | documents kept in memory; the oldest is evicted beyond this |
| `LLM_MODEL`, `EMBEDDING_MODEL` | LLaMA 3.3 70B, all-MiniLM-L6-v2 | models used |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | 500, 50, 3 | retrieval settings |

## Tests and CI

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check . && pytest
```

The tests use a fake LLM and fake embeddings, so they run in under a second with no API key and
no model download. They cover document isolation, upload validation (type, size, corrupt files,
hostile file names), temp-file cleanup, error codes, the eviction limit and the API key. They do
not test answer quality. GitHub Actions runs the same checks on every push.

## Limitations

- **Indexes live in memory.** Restarting the server loses them. Persisting FAISS to disk is next.
- **Text PDFs only.** Scanned documents without a text layer are rejected (no OCR).
- **Not evaluated.** There is no measurement yet of retrieval or answer quality.
- **Basic protection only.** One optional shared API key, no per-user accounts, no rate limiting.
- **Single process.** The in-memory store is not shared between workers or machines.
- `langchain-community` (used for the FAISS wrapper) is being sunset upstream; the vector store
  will need to move to a maintained integration.

## Roadmap

1. Evaluation set: questions with expected answers on sample PDFs, with the results published here.
2. Persist indexes to disk and reload them at startup.
3. Streaming answers and conversation history.

## Author

Didier Nathan Megnengue Akono, AI Engineer.
[LinkedIn](https://linkedin.com/in/NathanMegnengue) · [GitHub](https://github.com/NathanMegnengue)
