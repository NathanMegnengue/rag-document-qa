"""Measure retrieval and answer quality on the fictional handbook.

    python -m evaluation.run                   # full run, needs GROQ_API_KEY
    python -m evaluation.run --retrieval-only  # no LLM call: checks that the right chunks are found
    python -m evaluation.run --out evaluation/RESULTS-other.md \
        --embedding-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Answer correctness is checked with regular expressions on the expected facts (questions.json), so
a correct answer worded in an unexpected way counts as a miss: read the per-question table too.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, get_settings
from app.rag import RagService, build_service, page_of

HERE = Path(__file__).parent
HANDBOOK = HERE / "handbook.pdf"
QUESTIONS = HERE / "questions.json"


@dataclass
class QuestionResult:
    id: str
    question: str
    answerable: bool
    retrieved_pages: list[int]
    # answerable only: a retrieved chunk contains the evidence sentence (the real retrieval test)
    retrieval_hit: bool | None = None
    # answerable only: a retrieved chunk comes from an expected page (looser, shown for contrast)
    page_hit: bool | None = None
    answer: str | None = None
    found: bool | None = None
    answer_ok: bool | None = None  # answerable only: found and every expected fact present
    refusal_ok: bool | None = None  # unanswerable only: the API said "not found"
    source_pages: list[int] | None = None
    source_precision: float | None = None  # answerable only: share of cited pages that are expected


def _norm(text: str) -> str:
    return " ".join(text.split())


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(
    service: RagService,
    doc_id: str,
    questions: list[dict],
    *,
    retrieval_only: bool = False,
    delay: float = 0.0,
) -> list[QuestionResult]:
    results = []
    for q in questions:
        answerable = q["answerable"]
        expected_pages = set(q["expected_pages"])
        hits = service.retrieve(doc_id, q["question"])
        pages = sorted({page_of(d) for d in hits})
        res = QuestionResult(q["id"], q["question"], answerable, pages)
        if answerable:
            res.page_hit = bool(expected_pages & set(pages))
            evidence = _norm(q["evidence"])
            res.retrieval_hit = any(evidence in _norm(d.page_content) for d in hits)
        if not retrieval_only:
            ans = service.ask(doc_id, q["question"])
            res.answer, res.found = ans.answer, ans.found
            res.source_pages = [s.page for s in ans.sources]
            if answerable:
                res.answer_ok = ans.found and all(
                    re.search(rx, ans.answer, re.IGNORECASE) for rx in q["expected_regex"]
                )
                if res.source_pages:
                    good = sum(1 for p in res.source_pages if p in expected_pages)
                    res.source_precision = good / len(res.source_pages)
            else:
                res.refusal_ok = not ans.found
            if delay:
                time.sleep(delay)  # stay under the provider's rate limit
        results.append(res)
    return results


def _rate(values: list[bool | float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: list[QuestionResult]) -> dict:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    answered = [r for r in answerable if r.answer_ok is not None]
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "retrieval_hit_rate": _rate([bool(r.retrieval_hit) for r in answerable]),
        "page_hit_rate": _rate([bool(r.page_hit) for r in answerable]),
        "answer_accuracy": _rate([bool(r.answer_ok) for r in answered]),
        "false_refusal_rate": _rate([r.found is False for r in answered]),
        "correct_refusal_rate": _rate(
            [bool(r.refusal_ok) for r in unanswerable if r.refusal_ok is not None]
        ),
        "source_precision": _rate(
            [r.source_precision for r in answerable if r.source_precision is not None]
        ),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("✓" if value else "✗")


def render_markdown(summary: dict, results: list[QuestionResult], meta: dict) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"- Date (UTC): {meta['date']}",
        f"- Document: `evaluation/handbook.pdf` (fictional, {summary['n_answerable']} answerable "
        f"and {summary['n_unanswerable']} unanswerable questions, French and English)",
        f"- Embedding model: `{meta['embedding_model']}`",
        f"- LLM: `{meta['llm_model'] or 'not used (retrieval only)'}`",
        f"- Chunks: size {meta['chunk_size']}, overlap {meta['chunk_overlap']}, "
        f"top_k {meta['top_k']}",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Retrieval hit rate (a top-{meta['top_k']} chunk contains the evidence) "
        f"| {_pct(summary['retrieval_hit_rate'])} |",
        f"| Page hit rate (a top-{meta['top_k']} chunk is on the expected page; looser) "
        f"| {_pct(summary['page_hit_rate'])} |",
        f"| Answer accuracy | {_pct(summary['answer_accuracy'])} |",
        f"| False refusals (answerable questions refused) "
        f"| {_pct(summary['false_refusal_rate'])} |",
        f"| Correct refusals (unanswerable questions refused) "
        f"| {_pct(summary['correct_refusal_rate'])} |",
        f"| Source precision (cited pages that are the right ones) "
        f"| {_pct(summary['source_precision'])} |",
        "",
        "Answer accuracy is checked with regular expressions on the expected facts, so a correct "
        "answer worded unexpectedly counts as a miss. The sample is small: treat the numbers as "
        "indicative, not as a benchmark.",
        "",
        "| Id | Question | Retrieval | Page | Answer | Refusal | Cited pages |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cited = "-" if r.source_pages is None else (", ".join(map(str, r.source_pages)) or "none")
        lines.append(
            f"| {r.id} | {r.question} | {_mark(r.retrieval_hit)} | {_mark(r.page_hit)} "
            f"| {_mark(r.answer_ok)} "
            f"| {_mark(r.refusal_ok)} | {cited} |"
        )
    return "\n".join(lines) + "\n"


def build_eval_service(settings: Settings, retrieval_only: bool) -> RagService:
    if not retrieval_only:
        return build_service(settings)
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return RagService(embeddings, FakeListChatModel(responses=["unused"]), settings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM calls")
    parser.add_argument("--embedding-model", help="override EMBEDDING_MODEL for this run")
    parser.add_argument("--out", type=Path, default=HERE / "RESULTS.md")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between LLM calls")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.embedding_model:
        settings = dataclasses.replace(settings, embedding_model=args.embedding_model)

    service = build_eval_service(settings, args.retrieval_only)
    info = service.ingest(str(HANDBOOK), HANDBOOK.name)
    results = evaluate(
        service,
        info.id,
        load_questions(),
        retrieval_only=args.retrieval_only,
        delay=args.delay,
    )
    meta = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "embedding_model": settings.embedding_model,
        "llm_model": None if args.retrieval_only else settings.llm_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
    }
    report = render_markdown(summarize(results), results, meta)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
