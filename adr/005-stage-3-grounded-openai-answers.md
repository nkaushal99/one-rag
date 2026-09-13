# ADR 005: Generate answers only from visible retrieved context

## Status

Accepted

## Decision

Expose `POST /v1/answer`. It reuses the existing retrieval, neighbor-expansion,
and parent-context flow, builds the same deduplicated context returned by
`/v1/query`, then invokes Gemini through LangChain using
`gemini-2.5-flash-lite`. The prompt requires the
model to use only that context, identify missing evidence, and cite source/chunk
labels. The response includes the answer, context, and evidence.

`GOOGLE_API_KEY` is read from the ignored local `.env`; it is not added to the
template or API response. `LLM_PROVIDER`, `LLM_MODEL`, and `LLM_TEMPERATURE`
remain configurable. There is no answer CLI; HTTP endpoints are the supported
interface.

## Consequences

Generation is no longer deterministic, but its supporting evidence stays
inspectable. Endpoint tests mock the model; a separate live smoke test verifies
the configured Gemini key and model.
