# ADR 0002 — Streamlit for the UI

- Status: Accepted
- Date: 2026-04-20
- Supersedes: ADR 0001-draft (Next.js)

## Context

The UI's job is to expose every feature for hands-on demos to recruiters, teammates, and self-review. It is not a customer-facing product.

## Options considered

| Option       | Setup time | DX     | Demos well? | All-Python? |
| ------------ | ---------- | ------ | ----------- | ----------- |
| Next.js + TS | High       | Great  | Excellent   | No          |
| Gradio       | Low        | OK     | Good        | Yes         |
| Reflex       | Medium     | Good   | OK          | Yes         |
| **Streamlit**| Low        | Great  | Excellent   | Yes         |

## Decision

Use **Streamlit** as a multi-page app, talking to the API via HTTPX. Each feature is its own page in `services/ui/echomind_ui/pages/`.

## Consequences

- Total UI code drops from ~1500 lines (Next.js) to ~400 lines (Streamlit).
- The language stack is 100% Python — easier for ML reviewers to read end-to-end.
- Pixel-perfect customization is harder; out of scope.
- The UI only ever talks to the API — never the DB directly — which keeps the service boundary clean.
