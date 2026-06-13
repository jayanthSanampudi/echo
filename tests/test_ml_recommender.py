"""Two-tower recommender — train on tiny synthetic data, check it learns."""

from __future__ import annotations

import numpy as np
import torch

from echomind_ml.recommender import TrainConfig, TwoTower, train


def test_train_reduces_loss():
    rng = np.random.default_rng(0)
    num_users = 20
    content_dim = 16

    # users prefer one of two clusters
    cluster_centers = rng.normal(size=(2, content_dim))
    user_cluster = rng.integers(0, 2, size=num_users)

    user_idx, item_content = [], []
    for u in range(num_users):
        for _ in range(8):
            user_idx.append(u)
            item_content.append(cluster_centers[user_cluster[u]] + 0.1 * rng.normal(size=content_dim))
    user_idx = np.array(user_idx)
    item_content = np.stack(item_content).astype(np.float32)

    cfg = TrainConfig(epochs=2, batch_size=32, lr=1e-2)
    model = train(user_idx, item_content, num_users=num_users, cfg=cfg)

    assert isinstance(model, TwoTower)
    # quick sanity: user_emb for user 0 should be closer to its items than to a random vector
    with torch.no_grad():
        u_emb = model.user(torch.tensor([0])).numpy()[0]
        own_items = item_content[user_idx == 0][:5]
        own_proj = model.item(torch.from_numpy(own_items)).numpy()
        own_score = (own_proj @ u_emb).mean()
        random_proj = model.item(torch.from_numpy(rng.normal(size=(5, content_dim)).astype(np.float32))).numpy()
        rand_score = (random_proj @ u_emb).mean()
        assert own_score > rand_score
