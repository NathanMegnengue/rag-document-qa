"""Checks the evaluation dataset against the handbook, and the harness with fake models."""

import dataclasses

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pypdf import PdfReader

from app.config import Settings
from app.rag import RagService
from evaluation.run import HANDBOOK, evaluate, load_questions, render_markdown, summarize


def page_texts() -> list[str]:
    return [" ".join(p.extract_text().split()) for p in PdfReader(HANDBOOK).pages]


def test_dataset_is_consistent_with_the_handbook():
    questions, pages = load_questions(), page_texts()
    assert len({q["id"] for q in questions}) == len(questions)
    assert sum(q["answerable"] for q in questions) >= 15
    assert sum(not q["answerable"] for q in questions) >= 3
    for q in questions:
        if q["answerable"]:
            assert q["expected_regex"], q["id"]
            for page in q["expected_pages"]:
                assert 1 <= page <= len(pages), q["id"]
                assert q["evidence"] in pages[page - 1], f"{q['id']}: evidence not on page {page}"
        else:
            assert not q["expected_pages"] and not q["expected_regex"], q["id"]


def test_every_evidence_sentence_fits_inside_one_chunk():
    """Otherwise the chunk-level retrieval metric could never be satisfied for that question."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    settings = Settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    pages = [Document(page_content=p.extract_text()) for p in PdfReader(HANDBOOK).pages]
    chunks = [" ".join(c.page_content.split()) for c in splitter.split_documents(pages)]
    for q in load_questions():
        if q["answerable"]:
            assert any(q["evidence"] in c for c in chunks), q["id"]


@pytest.fixture
def service_factory():
    def make(response: str) -> RagService:
        settings = Settings(top_k=3)
        return RagService(
            DeterministicFakeEmbedding(size=32), FakeListChatModel(responses=[response]), settings
        )

    return make


def run(service, retrieval_only):
    info = service.ingest(str(HANDBOOK), "handbook.pdf")
    return evaluate(service, info.id, load_questions(), retrieval_only=retrieval_only)


def test_a_model_that_always_refuses_is_scored_accordingly(service_factory):
    summary = summarize(run(service_factory("NOT_FOUND\nNope."), retrieval_only=False))
    assert summary["correct_refusal_rate"] == 1.0
    assert summary["false_refusal_rate"] == 1.0
    assert summary["answer_accuracy"] == 0.0


def test_a_model_that_never_gives_the_facts_has_zero_accuracy(service_factory):
    summary = summarize(run(service_factory("Something vague."), retrieval_only=False))
    assert summary["answer_accuracy"] == 0.0
    assert summary["false_refusal_rate"] == 0.0
    assert summary["correct_refusal_rate"] == 0.0  # it never refuses, even when it should


def test_retrieval_only_mode_makes_no_llm_metrics(service_factory):
    results = run(service_factory("unused"), retrieval_only=True)
    summary = summarize(results)
    assert summary["retrieval_hit_rate"] is not None and summary["page_hit_rate"] is not None
    assert summary["retrieval_hit_rate"] <= summary["page_hit_rate"]
    assert summary["answer_accuracy"] is None and summary["correct_refusal_rate"] is None
    assert all(r.answer is None for r in results)


def test_report_is_rendered(service_factory):
    results = run(service_factory("NOT_FOUND"), retrieval_only=False)
    meta = {
        **dataclasses.asdict(Settings(top_k=3)),
        "date": "2026-01-01 00:00",
        "llm_model": "fake",
    }
    report = render_markdown(summarize(results), results, meta)
    assert "Retrieval hit rate" in report and "| q01 |" in report
