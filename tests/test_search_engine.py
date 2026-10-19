"""Tests unitaires validant la normalisation L2, la similarité cosinus et l'indexation FAISS."""

import numpy as np
from src.embedder import l2_normalize
from src.search_engine import MultimodalSearchEngine


def test_l2_normalization_unit_norm():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 512)).astype(np.float32)
    X_norm = l2_normalize(X)
    norms = np.linalg.norm(X_norm, axis=1)
    np.testing.assert_allclose(norms, np.ones(20), atol=1e-5)


def test_exact_numpy_vs_faiss_flat_equivalence():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(500, 128)).astype(np.float32)
    query = rng.normal(size=(128,)).astype(np.float32)

    engine = MultimodalSearchEngine(embeddings)
    scores_np, idx_np, _ = engine.search_exact_numpy(query, top_k=5)
    scores_faiss, idx_faiss, _ = engine.search_faiss_flat(query, top_k=5)

    np.testing.assert_array_equal(idx_np, idx_faiss)
    np.testing.assert_allclose(scores_np, scores_faiss, atol=1e-5)


def test_faiss_hnsw_returns_valid_top_k():
    rng = np.random.default_rng(123)
    embeddings = rng.normal(size=(1000, 64)).astype(np.float32)
    query = embeddings[10]  # On cherche un vecteur qui existe exactement dans la base

    engine = MultimodalSearchEngine(embeddings)
    scores_hnsw, idx_hnsw, latency = engine.search_faiss_hnsw(query, top_k=5)

    assert len(idx_hnsw) == 5
    assert idx_hnsw[0] == 10
    assert np.isclose(scores_hnsw[0], 1.0, atol=1e-5)
    assert latency >= 0.0