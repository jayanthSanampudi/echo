# ADR 0003 — Recommender architecture

- Status: Accepted
- Date: 2026-04-22

## Context

We need a recommender that:

- Trains on a laptop in seconds for the demo.
- Handles cold-start books (no listening history) gracefully.
- Maps to a real production architecture so it's defensible in interviews.

## Options considered

1. **Popularity** — trivial baseline; no personalization.
2. **Matrix factorization (ALS)** — strong with sufficient data, but cold-start is hard.
3. **Two-tower deep model** — what YouTube, TikTok, and most modern recs use.
4. **GNN** — overkill for the catalog sizes we'll demo.

## Decision

**Two-tower** with:

- User tower: ID embedding → MLP → 128-dim
- Item tower: content embedding (text+audio fusion) → MLP → 128-dim
- Loss: in-batch sampled softmax (scales to billions of items)
- Inference: precompute item embeddings, store in a numpy array; user lookup is a matrix-vector product

Cold-start books are handled because the item tower takes content as input — no item-side embedding table to populate.

## Consequences

- The same architecture and training script we ship would scale to production with the only changes being: (a) real interaction data, (b) a sharded item index, (c) Triton / TorchServe for inference.
- The model is small (<10 MB) and trains in seconds on CPU.
- Eval uses hit@k and nDCG; reproducible via `notebooks/03_recommender_training.ipynb`.
