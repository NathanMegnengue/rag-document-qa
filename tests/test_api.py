import os

from tests.conftest import make_pdf


def upload(client, pages, name="doc.pdf", **kw):
    return client.post(
        "/documents", files={"file": (name, make_pdf(pages), "application/pdf")}, **kw
    )


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_returns_document_info(client):
    r = upload(client, ["Alpha page one", "Alpha page two"])
    assert r.status_code == 201
    body = r.json()
    assert body["pages"] == 2 and body["chunks"] >= 2 and body["filename"] == "doc.pdf"
    assert len(body["id"]) == 32


def test_ask_returns_answer_and_sources_with_page_numbers(client):
    doc = upload(client, ["Alpha page one", "Alpha page two"]).json()
    r = client.post(f"/documents/{doc['id']}/ask", json={"question": "What is this about?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "The answer is 42."
    assert {s["page"] for s in body["sources"]} == {1, 2}
    assert all("Alpha" in s["excerpt"] for s in body["sources"])


def test_documents_are_isolated(client):
    a = upload(client, ["alpha only"], name="a.pdf").json()
    b = upload(client, ["beta only"], name="b.pdf").json()
    sources_a = client.post(f"/documents/{a['id']}/ask", json={"question": "q"}).json()["sources"]
    sources_b = client.post(f"/documents/{b['id']}/ask", json={"question": "q"}).json()["sources"]
    assert all("alpha" in s["excerpt"] for s in sources_a)
    assert all("beta" in s["excerpt"] for s in sources_b)


def test_ask_unknown_document_is_404(client):
    r = client.post("/documents/doesnotexist/ask", json={"question": "hello"})
    assert r.status_code == 404


def test_empty_question_is_rejected(client):
    doc = upload(client, ["text"]).json()
    r = client.post(f"/documents/{doc['id']}/ask", json={"question": ""})
    assert r.status_code == 422


def test_non_pdf_is_rejected(client):
    r = client.post("/documents", files={"file": ("x.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 400


def test_too_large_file_is_rejected(client):
    big = b"%PDF-" + b"0" * (2 * 1024 * 1024)
    r = client.post("/documents", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413


def test_corrupt_pdf_is_422(client):
    files = {"file": ("bad.pdf", b"%PDF-1.4 garbage", "application/pdf")}
    assert client.post("/documents", files=files).status_code == 422


def test_no_temp_file_is_left_behind(client, settings):
    upload(client, ["text"])
    client.post("/documents", files={"file": ("x.pdf", b"nope", "application/pdf")})
    client.post("/documents", files={"file": ("bad.pdf", b"%PDF-1.4 garbage", "application/pdf")})
    assert os.listdir(settings.tmp_dir) == []


def test_malicious_filename_is_only_display_text(client, settings):
    r = upload(client, ["text"], name="../../etc/evil.pdf")
    assert r.status_code == 201
    assert r.json()["filename"] == "evil.pdf"


def test_oldest_document_is_evicted_beyond_the_limit(client):
    ids = [upload(client, [f"doc {i}"], name=f"{i}.pdf").json()["id"] for i in range(3)]
    assert client.post(f"/documents/{ids[0]}/ask", json={"question": "q"}).status_code == 404
    assert client.post(f"/documents/{ids[2]}/ask", json={"question": "q"}).status_code == 200


def test_list_and_delete(client):
    doc = upload(client, ["text"]).json()
    assert [d["id"] for d in client.get("/documents").json()] == [doc["id"]]
    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_api_key_is_enforced_when_configured(make_client):
    client = make_client(api_key="secret")
    assert client.get("/health").status_code == 200
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "secret"}).status_code == 200
