# RAG Document QA

A production-ready REST API that lets you upload any PDF and ask questions about it in natural language.

Built with LangChain, LLaMA 3 (via Groq), FAISS, and FastAPI — deployed locally via Uvicorn.

---

## What it does

You upload a PDF. The system splits it into chunks, vectorizes them using a sentence-transformer embedding model, and stores them in a FAISS vector index. When you ask a question, it retrieves the most relevant chunks and passes them to LLaMA 3 to generate a grounded answer.

No hallucination from thin air — the model only answers based on what's in your document.

---

## Architecture
PDF upload
│
▼
PyPDFLoader → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
│
▼
HuggingFace Embeddings (all-MiniLM-L6-v2)
│
▼
FAISS Vector Store
│
▼
Retriever (top-3 chunks) → LLaMA 3.3 70B via Groq API
│
▼
Answer

---

## Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| LLM | LLaMA 3.3 70B (Groq) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| PDF Loader | PyPDFLoader (LangChain) |
| Orchestration | LangChain LCEL |
| Server | Uvicorn |

---

## Getting started

**1. Clone the repo**
```bash
git clone https://github.com/NathanMegnengue/rag-document-qa.git
cd rag-document-qa
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your Groq API key**

Create a `.env` file at the root:
GROQ_API_KEY=your_groq_api_key_here
Get a free key at [console.groq.com](https://console.groq.com)

**5. Run the API**
```bash
uvicorn app.main:app --reload
```

API is live at `http://127.0.0.1:8000`

---

## Usage

**Upload a PDF**
```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@your_document.pdf"
```

**Ask a question**
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

**Interactive docs**

Open `http://127.0.0.1:8000/docs` in your browser — FastAPI generates a full Swagger UI automatically.

---

## Project structure
rag-document-qa/
├── app/
│   ├── main.py       # FastAPI routes and entry point
│   └── rag.py        # RAG pipeline logic
├── .env              # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md

---

## Limitations and next steps

- The vector store is in-memory — restarting the server loses the index. Next step: persist FAISS to disk.
- Single document at a time. Next step: support multi-document ingestion.
- No authentication on the API endpoints. Next step: add API key middleware.

---

## Author

**Didier Nathan Megnengue Akono** — AI Engineer  
[LinkedIn](https://linkedin.com/in/NathanMegnengue) · [GitHub](https://github.com/NathanMegnengue)