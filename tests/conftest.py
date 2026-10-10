import dataclasses

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings, get_settings
from app.main import app, get_service
from app.rag import RagService


def make_pdf(pages: list[str]) -> bytes:
    """Build a tiny valid PDF with one line of text per page (no extra dependency needed)."""
    objects: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(
            b"<< /Length "
            + str(len(content)).encode()
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    size = len(objects) + 1
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(tmp_dir=str(tmp_path), max_upload_mb=1, max_documents=2)


@pytest.fixture
def make_client(settings):
    def _make(llm=None, **overrides) -> TestClient:
        s = dataclasses.replace(settings, **overrides)
        service = RagService(
            embeddings=DeterministicFakeEmbedding(size=32),
            llm=llm or FakeListChatModel(responses=["The answer is 42."]),
            settings=s,
        )
        app.dependency_overrides[get_settings] = lambda: s
        app.dependency_overrides[get_service] = lambda: service
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
