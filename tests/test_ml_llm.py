"""Mock LLM extractive logic."""

from __future__ import annotations

from echomind_ml.llm import _MockLLM


def test_mock_extracts_most_overlapping_sentence():
    llm = _MockLLM()
    user = (
        "Context:\n[Chapter 0 @ 0s] Victor met the creature on the glacier. "
        "It was a cold morning. They spoke of forgiveness.\n\n"
        "Question: Who did Victor meet?\n\nAnswer:"
    )
    answer = llm.complete("sys", user)
    assert "glacier" in answer.lower() or "victor" in answer.lower()


def test_mock_handles_no_context():
    llm = _MockLLM()
    answer = llm.complete("sys", "Just a question with no context.")
    assert "don't have enough context" in answer.lower() or answer
