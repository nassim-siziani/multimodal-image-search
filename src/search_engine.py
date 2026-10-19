"""Moteur de recherche vectorielle comparant le calcul Cosinus exact (NumPy) et l'indexation FAISS."""

import time
import faiss
import numpy as np
from src.embedder import l2_normalize


class MultimodalSearchEngine:
    """
    Gère l'indexation d'une base de vecteurs d'images et compare trois méthodes de recherche :
    1. Exact Cosine (NumPy) : Calcul matriciel exhaustif O(N * d).
    2. FAISS FlatIP : Recherche exacte par produit scalaire optimisée en C++ / SIMD.
    3. FAISS HNSW : Recherche approchée (ANN) par graphe hiérarchique (Hierarchical Navigable Small World).
    """

    def __init__(self, embeddings: np.ndarray, hnsw_m: int = 16):
        self.embeddings = l2_normalize(embeddings)
        self.num_samples, self.dim = self.embeddings.shape

        # 1. Index FAISS exact (Inner Product sur vecteurs normalisés L2 = Similarité Cosinus)
        self.index_flat = faiss.IndexFlatIP(self.dim)
        self.index_flat.add(self.embeddings)

        # 2. Index FAISS approché HNSW (Graphe de proximité pour grandes bases)
        self.index_hnsw = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self.index_hnsw.hnsw.efConstruction = 40
        self.index_hnsw.hnsw.efSearch = 32
        self.index_hnsw.add(self.embeddings)

    def search_exact_numpy(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
        """Recherche exacte par produit matriciel NumPy."""
        q = l2_normalize(query_vector.reshape(1, -1))
        t0 = time.perf_counter()

        # Similarité cosinus = produit scalaire car ||q||_2 = 1 et ||v||_2 = 1
        similarities = np.dot(self.embeddings, q.T).squeeze(axis=1)
        top_k = min(top_k, self.num_samples)
        indices = np.argsort(-similarities)[:top_k]
        scores = similarities[indices]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return scores, indices, latency_ms

    def search_faiss_flat(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
        """Recherche exacte accélérée via FAISS IndexFlatIP."""
        q = l2_normalize(query_vector.reshape(1, -1))
        top_k = min(top_k, self.num_samples)
        t0 = time.perf_counter()

        scores, indices = self.index_flat.search(q, top_k)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return scores[0], indices[0], latency_ms

    def search_faiss_hnsw(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray, float]:
        """Recherche approchée rapide par graphe via FAISS IndexHNSWFlat."""
        q = l2_normalize(query_vector.reshape(1, -1))
        top_k = min(top_k, self.num_samples)
        t0 = time.perf_counter()

        scores, indices = self.index_hnsw.search(q, top_k)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return scores[0], indices[0], latency_ms

    def search(self, query_vector: np.ndarray, top_k: int = 5, method: str = "faiss_flat") -> tuple[np.ndarray, np.ndarray, float]:
        """Point d'entrée unifié permettant de choisir l'algorithme de recherche."""
        if method == "numpy":
            return self.search_exact_numpy(query_vector, top_k=top_k)
        if method == "faiss_flat":
            return self.search_faiss_flat(query_vector, top_k=top_k)
        if method == "faiss_hnsw":
            return self.search_faiss_hnsw(query_vector, top_k=top_k)
        raise ValueError(f"Méthode inconnue : {method}. Choisir parmi 'numpy', 'faiss_flat', 'faiss_hnsw'.")