"""Évaluation de la pertinence (Precision@K) et benchmark de vitesse (NumPy vs FAISS)."""

import argparse
import numpy as np
from src.dataset import prepare_demo_dataset
from src.embedder import CLIPEmbedder
from src.search_engine import MultimodalSearchEngine


TEST_QUERIES = {
    "airplane": "a commercial airplane flying in the sky",
    "automobile": "a red car parked on the street",
    "bird": "a wild bird with feathers",
    "cat": "a domestic cat looking at the camera",
    "deer": "a wild deer in the forest",
    "dog": "a cute dog puppy",
    "frog": "a green frog in nature",
    "horse": "a horse running in a field",
    "ship": "a large boat or cargo ship on the sea",
    "truck": "a heavy delivery truck on the road",
}


def evaluate_retrieval_quality(num_images: int = 200):
    """Mesure la Precision@1 et la Precision@5 de CLIP sur les requêtes de test."""
    print("=== 1. Évaluation de la Pertinence Sémantique (CLIP) ===")
    image_paths, labels = prepare_demo_dataset(num_images=num_images)
    labels_arr = np.array(labels)

    embedder = CLIPEmbedder()
    print("Encodage des images avec CLIP...")
    img_embeddings = embedder.embed_images(image_paths)
    engine = MultimodalSearchEngine(img_embeddings)

    p_at_1_list = []
    p_at_5_list = []

    print("\nRésultats par requête textuelle :")
    for target_label, query_text in TEST_QUERIES.items():
        q_vec = embedder.embed_texts([query_text])[0]
        _, indices, _ = engine.search_faiss_flat(q_vec, top_k=5)
        retrieved_labels = labels_arr[indices]

        p_at_1 = 1.0 if retrieved_labels[0] == target_label else 0.0
        p_at_5 = float(np.mean(retrieved_labels == target_label))

        p_at_1_list.append(p_at_1)
        p_at_5_list.append(p_at_5)
        print(f"  - Requête : '{query_text}' -> Precision@1: {p_at_1*100:5.1f}% | Precision@5: {p_at_5*100:5.1f}%")

    print(f"\nMoyenne Globale -> Precision@1 : {np.mean(p_at_1_list)*100:.1f}% | Precision@5 : {np.mean(p_at_5_list)*100:.1f}%\n")


def run_scaling_benchmark(sizes: list[int] | None = None, dim: int = 512, n_queries: int = 50):
    """Compare le temps de recherche (latence moyenne en ms) selon la taille de la base N."""
    if sizes is None:
        sizes = [1_000, 10_000, 50_000, 100_000]

    print("=== 2. Benchmark Algorithmique de Passage à l'Échelle (NumPy vs FAISS) ===")
    print(f"{'Taille Base (N)':<18} | {'NumPy Cosinus (ms)':<20} | {'FAISS FlatIP (ms)':<20} | {'FAISS HNSW (ms)':<18} | {'Rappel@5 HNSW':<15}")
    print("-" * 98)

    rng = np.random.default_rng(42)

    for n in sizes:
        db_vectors = rng.standard_normal((n, dim), dtype=np.float32)
        queries = rng.standard_normal((n_queries, dim), dtype=np.float32)
        engine = MultimodalSearchEngine(db_vectors)

        lat_numpy, lat_flat, lat_hnsw = [], [], []
        hnsw_recalls = []

        for q in queries:
            _, idx_np, t_np = engine.search_exact_numpy(q, top_k=5)
            _, idx_fl, t_fl = engine.search_faiss_flat(q, top_k=5)
            _, idx_hn, t_hn = engine.search_faiss_hnsw(q, top_k=5)

            lat_numpy.append(t_np)
            lat_flat.append(t_fl)
            lat_hnsw.append(t_hn)
            # Proportion des vrais 5 plus proches voisins retrouvés par l'approximation HNSW
            overlap = len(set(idx_np).intersection(set(idx_hn))) / 5.0
            hnsw_recalls.append(overlap)

        print(
            f"{n:<18,d} | {np.mean(lat_numpy):<20.4f} | {np.mean(lat_flat):<20.4f} | "
            f"{np.mean(lat_hnsw):<18.4f} | {np.mean(hnsw_recalls)*100:6.1f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation et benchmark du moteur de recherche CLIP + FAISS.")
    parser.add_argument("--num-images", type=int, default=200, help="Nombre d'images pour l'évaluation CLIP")
    parser.add_argument("--skip-clip", action="store_true", help="Lancer uniquement le benchmark vectoriel FAISS")
    args = parser.parse_args()

    if not args.skip_clip:
        evaluate_retrieval_quality(num_images=args.num_images)
    run_scaling_benchmark()