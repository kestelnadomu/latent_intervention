"""Synthetic check of the fact marking in the decoder appendix example."""

from exp.paper_tables.decoder_example import marked

CODEBOOK = {"choices": {"X": {0: ["Kenya"], 3: ["Spain", "the Netherlands"]}}}


def test_facts_marked_against_the_reference():
    reference = "I am from Spain, hold a Master's degree from a global top 100 university and completed 5 projects."
    text = "I am from the Netherlands, hold a Bachelor's degree from a global top 100 university & completed 5 projects."
    tex = marked(text, reference, CODEBOOK)
    assert r"\factright{Netherlands}" in tex  # same region as Spain
    assert r"\factwrong{Bachelor's}" in tex
    assert r"\factright{global top 100}" in tex and r"\factright{5 projects}" in tex
    assert r"university \& completed" in tex
    assert r"\factwrong{Kenya}" in marked("I am from Kenya.", reference, CODEBOOK)
