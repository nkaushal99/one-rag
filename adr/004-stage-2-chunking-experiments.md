# ADR 004: Compare chunk boundaries before selecting a default

## Status

Accepted

## Context

A relevant answer can span adjacent sentences or paragraphs. Fixed-size chunks
can split that answer, while broad structural chunks can retrieve too much
unrelated context. Cosine similarity is a ranking signal, not an accuracy
measurement.

## Decision

Implement six inspectable strategies: 500 lexical-token fixed chunks, 500-token
fixed chunks with 100-token overlap, sentence chunks, two-sentence windows with
one-sentence overlap, paragraphs, and Markdown-heading sections. Run
`scripts/evaluate_chunking.py` against the same documents and queries for every
strategy. Record chunk counts, top scores, retrieved chunk indexes, and whether
the top-three context includes every expected answer phrase.

The default remains `sentence_window` until the measured corpus says otherwise.
The evaluator retains separate Qdrant collections named `chunking_eval_*` so the
retrieved evidence remains inspectable.

For the larger enterprise handbook, `parent_child` keeps each structural section
as a parent while indexing overlapping two-sentence child windows. Optional
neighbor retrieval expands a matched child only within its parent; optional
parent context returns the full section after a child match.

## Consequences

The experiment exposes answer completeness as well as ranking. The simple
sentence-window strategy is a context-expansion baseline; parent-child chunks
and neighbor retrieval remain follow-up strategies if completeness is still
poor.
