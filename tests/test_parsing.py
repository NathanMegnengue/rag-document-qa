import pytest

from app.rag import NOT_FOUND_ANSWER, parse_model_output


def test_plain_answer_without_used_line_keeps_all_sources():
    parsed = parse_model_output("Paris is the capital.", 3)
    assert parsed.found and parsed.text == "Paris is the capital." and parsed.used is None


def test_used_line_is_removed_and_parsed():
    parsed = parse_model_output("The budget is 1500 euros.\nUSED: 3, 1", 3)
    assert parsed.found
    assert parsed.text == "The budget is 1500 euros."
    assert parsed.used == [1, 3]


@pytest.mark.parametrize("line", ["used: 2", "**USED:** 2", "USED : 2 "])
def test_used_line_is_tolerant_to_formatting(line):
    parsed = parse_model_output(f"Answer.\n{line}", 3)
    assert parsed.text == "Answer." and parsed.used == [2]


def test_used_numbers_outside_the_passages_are_ignored():
    parsed = parse_model_output("Answer.\nUSED: 7, 2", 3)
    assert parsed.used == [2]


def test_used_none_falls_back_to_all_sources():
    parsed = parse_model_output("Answer.\nUSED: none", 3)
    assert parsed.found and parsed.used is None


def test_not_found_keeps_the_explanation_in_the_question_language():
    parsed = parse_model_output("NOT_FOUND\nLe document ne contient pas cette information.", 3)
    assert not parsed.found
    assert parsed.text == "Le document ne contient pas cette information."
    assert parsed.used == []


@pytest.mark.parametrize("raw", ["NOT_FOUND", "not_found.", "**NOT_FOUND**", NOT_FOUND_ANSWER])
def test_not_found_without_explanation_uses_the_default_sentence(raw):
    parsed = parse_model_output(raw, 3)
    assert not parsed.found and parsed.text == NOT_FOUND_ANSWER


def test_not_found_on_the_same_line():
    parsed = parse_model_output("NOT_FOUND: nothing about this.", 3)
    assert not parsed.found and parsed.text == "nothing about this."


def test_an_answer_that_merely_mentions_not_found_is_still_an_answer():
    parsed = parse_model_output("The file was not found on page 2.\nUSED: 1", 3)
    assert parsed.found


def test_empty_answer_counts_as_not_found():
    parsed = parse_model_output("USED: 1", 3)
    assert not parsed.found and parsed.text == NOT_FOUND_ANSWER
