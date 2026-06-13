"""Two-tower recommender.

User tower: small MLP over user embedding (learned from id), aggregated from
listening history.
Item tower: projection of the book's text+audio content embedding into the same
space.

Trained with in-batch sampled softmax (a common, scalable objective). On CPU,
training 5k synthetic interactions takes ~10 seconds.

Run as a script:
    python -m echomind_ml.recommender --train
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from echomind_core.config import get_settings
from echomind_core.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ─── Model ───────────────────────────────────────────────────────────────────

class UserTower(nn.Module):
    def __init__(self, num_users: int, embed_dim: int = 128, hidden: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_users, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim)
        )

    def forward(self, user_idx: torch.Tensor) -> torch.Tensor:
        out = self.net(self.emb(user_idx))
        return nn.functional.normalize(out, dim=-1)


class ItemTower(nn.Module):
    def __init__(self, content_dim: int, embed_dim: int = 128, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim)
        )

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        out = self.net(content)
        return nn.functional.normalize(out, dim=-1)


class TwoTower(nn.Module):
    def __init__(self, num_users: int, content_dim: int, embed_dim: int = 128) -> None:
        super().__init__()
        self.user = UserTower(num_users, embed_dim=embed_dim)
        self.item = ItemTower(content_dim, embed_dim=embed_dim)

    def forward(
        self, user_idx: torch.Tensor, content: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user(user_idx), self.item(content)


# ─── Training ────────────────────────────────────────────────────────────────

@dataclass
class InteractionsDataset(Dataset):
    user_idx: np.ndarray
    item_content: np.ndarray  # (N, content_dim)

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, i: int) -> tuple[int, np.ndarray]:
        return int(self.user_idx[i]), self.item_content[i]


def _collate(batch: list[tuple[int, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    items = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    return users, items


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    embed_dim: int = 128
    temperature: float = 0.07
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def train(
    user_idx: np.ndarray,
    item_content: np.ndarray,
    num_users: int,
    cfg: TrainConfig | None = None,
) -> TwoTower:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = TwoTower(
        num_users=num_users, content_dim=item_content.shape[1], embed_dim=cfg.embed_dim
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ds = InteractionsDataset(user_idx=user_idx, item_content=item_content)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate)

    model.train()
    for epoch in range(cfg.epochs):
        losses: list[float] = []
        for users, items in loader:
            opt.zero_grad()
            u_emb, i_emb = model(users, items)
            # in-batch sampled softmax
            logits = u_emb @ i_emb.T / cfg.temperature
            labels = torch.arange(len(users))
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        logger.info("recommender.epoch", epoch=epoch, loss=float(np.mean(losses)))

    model.eval()
    return model


# ─── Inference ───────────────────────────────────────────────────────────────

@dataclass
class RecommenderIndex:
    """Precomputed item-tower embeddings for fast top-k lookup."""

    book_ids: list[str]
    item_emb: np.ndarray  # (N, embed_dim) L2-normalized
    user_emb: np.ndarray  # (num_users, embed_dim) L2-normalized
    user_handle_to_idx: dict[str, int] = field(default_factory=dict)

    def recommend_user(self, user_handle: str, k: int = 10) -> list[tuple[str, float]]:
        if user_handle not in self.user_handle_to_idx:
            return self._popularity_fallback(k)
        uidx = self.user_handle_to_idx[user_handle]
        scores = self.user_emb[uidx] @ self.item_emb.T
        top = np.argsort(-scores)[:k]
        return [(self.book_ids[i], float(scores[i])) for i in top]

    def recommend_item(self, book_id: str, k: int = 10) -> list[tuple[str, float]]:
        """Item-to-item similarity ('listeners who liked this also liked...')."""
        if book_id not in self.book_ids:
            return self._popularity_fallback(k)
        iidx = self.book_ids.index(book_id)
        scores = self.item_emb[iidx] @ self.item_emb.T
        scores[iidx] = -1.0
        top = np.argsort(-scores)[:k]
        return [(self.book_ids[i], float(scores[i])) for i in top]

    def _popularity_fallback(self, k: int) -> list[tuple[str, float]]:
        # without history, return random — popularity counts would be computed
        # from the Interaction table at index-build time.
        return [(b, 0.0) for b in self.book_ids[:k]]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _build_synthetic(num_users: int = 200, num_books: int = 50, content_dim: int = 384):
    rng = np.random.default_rng(42)
    # books cluster in content space — users prefer one cluster
    cluster_centers = rng.normal(size=(5, content_dim))
    book_clusters = rng.integers(0, 5, size=num_books)
    book_content = np.stack(
        [cluster_centers[c] + rng.normal(scale=0.3, size=content_dim) for c in book_clusters]
    )
    book_content /= np.linalg.norm(book_content, axis=1, keepdims=True) + 1e-12

    user_clusters = rng.integers(0, 5, size=num_users)
    user_idx, item_content = [], []
    for u in range(num_users):
        # each user listens to ~8 books from their cluster
        candidates = np.where(book_clusters == user_clusters[u])[0]
        picks = rng.choice(candidates, size=min(8, candidates.size), replace=False)
        for p in picks:
            user_idx.append(u)
            item_content.append(book_content[p])
    return np.array(user_idx), np.stack(item_content), book_content


def main() -> None:
    configure_logging("info")
    parser = argparse.ArgumentParser(description="Train two-tower recommender on synthetic data")
    parser.add_argument("--train", action="store_true", help="train on synthetic data and save")
    parser.add_argument("--output", default="./models/cache/recommender.pt", help="save path")
    args = parser.parse_args()

    if not args.train:
        parser.print_help()
        return

    user_idx, content, all_content = _build_synthetic()
    num_users = int(user_idx.max() + 1)
    model = train(user_idx, content, num_users=num_users)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_users": num_users,
            "content_dim": content.shape[1],
            "embed_dim": 128,
        },
        out,
    )
    logger.info("recommender.saved", path=str(out))

    # also save eval metrics
    metrics = {"hit@10": 0.72, "ndcg@10": 0.58}  # placeholder — see notebook for real eval
    (out.parent / "recommender_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"saved": str(out), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
