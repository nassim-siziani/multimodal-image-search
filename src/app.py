"""Application Web Streamlit pour le moteur de recherche d'images multimodal CLIP & FAISS."""

from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

from src.dataset import prepare_demo_dataset
from src.embedder import CLIPEmbedder
from src.search_engine import MultimodalSearchEngine


st.set_page_config(page_title="Moteur de Recherche d'Images CLIP & FAISS", layout="wide")


@st.cache_resource
def load_resources(num_images: int = 200):
    """Charge le modèle CLIP, prépare les images et construit les index vectoriels en cache."""
    image_paths, labels = prepare_demo_dataset(num_images=num_images)
    embedder = CLIPEmbedder()

    cache_file = Path(f"data/embeddings_{num_images}.npz")
    if cache_file.exists():
        embeddings = np.load(cache_file)["embeddings"]
    else:
        embeddings = embedder.embed_images(image_paths)
        np.savez_compressed(cache_file, embeddings=embeddings)

    engine = MultimodalSearchEngine(embeddings)
    return embedder, engine, image_paths, labels


st.title("🔍 Moteur de Recherche Intelligent d'Images (CLIP & FAISS)")
st.markdown(
    "Recherchez des images en **langage naturel** ou **par similarité visuelle** grâce aux représentations "
    "multimodales de **CLIP (Vision Transformer)** et à l'indexation vectorielle **FAISS**."
)

embedder, engine, image_paths, labels = load_resources(num_images=200)

with st.sidebar:
    st.header("⚙️ Paramètres de recherche")
    method_label = st.selectbox(
        "Algorithme de recherche",
        options=[
            ("faiss_flat", "FAISS IndexFlatIP (Exact C++)"),
            ("faiss_hnsw", "FAISS IndexHNSW (Graphe Approché)"),
            ("numpy", "NumPy Cosinus (Produit Matriciel)"),
        ],
        format_func=lambda x: x[1],
    )
    selected_method = method_label[0]
    top_k = st.slider("Nombre de résultats (Top-K)", min_value=1, max_value=10, value=5)
    st.info(f"Base active : **{len(image_paths)} images** vectorisées en dimension **512**.")

tab_text, tab_image, tab_bench = st.tabs([
    "📝 Recherche par Texte",
    "🖼️ Recherche par Image",
    "⚡ Benchmark Algorithmique (NumPy vs FAISS)"
])

with tab_text:
    st.subheader("Décrivez la scène recherchée")
    query_text = st.text_input(
        "Requête en langage naturel (anglais ou français) :",
        value="a red car parked in the street",
    )
    if st.button("Rechercher par Texte", type="primary"):
        q_vec = embedder.embed_texts([query_text])[0]
        scores, indices, latency_ms = engine.search(q_vec, top_k=top_k, method=selected_method)

        st.success(f"Recherche effectuée en **{latency_ms:.3f} ms** avec `{selected_method}`.")
        cols = st.columns(min(top_k, 5))
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            with cols[rank % len(cols)]:
                st.image(image_paths[idx], use_container_width=True)
                st.caption(f"**#{rank+1}** | Catégorie : `{labels[idx]}`\nSimilarité cosinus : `{score:.3f}`")

with tab_image:
    st.subheader("Trouvez des images visuellement similaires")
    uploaded_file = st.file_uploader("Téléversez une image (.png, .jpg)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.image(query_img, caption="Image requête", width=200)
        q_vec = embedder.embed_images([query_img])[0]
        scores, indices, latency_ms = engine.search(q_vec, top_k=top_k, method=selected_method)

        st.success(f"Recherche par image effectuée en **{latency_ms:.3f} ms**.")
        cols = st.columns(min(top_k, 5))
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            with cols[rank % len(cols)]:
                st.image(image_paths[idx], use_container_width=True)
                st.caption(f"**#{rank+1}** | `{labels[idx]}` (`{score:.3f}`)")

with tab_bench:
    st.subheader("Comparaison des temps de recherche sur grande base vectorielle")
    n_sim = st.select_slider(
        "Taille de la base de vecteurs simulée (dimension 512) :",
        options=[1_000, 10_000, 50_000, 100_000],
        value=50_000,
    )
    if st.button("Lancer le Benchmark de vitesse"):
        with st.spinner(f"Génération et indexation de {n_sim:,} vecteurs..."):
            rng = np.random.default_rng(42)
            sim_db = rng.standard_normal((n_sim, 512), dtype=np.float32)
            sim_q = rng.standard_normal((1, 512), dtype=np.float32)[0]
            sim_engine = MultimodalSearchEngine(sim_db)

            _, idx_np, t_np = sim_engine.search_exact_numpy(sim_q, top_k=5)
            _, idx_fl, t_fl = sim_engine.search_faiss_flat(sim_q, top_k=5)
            _, idx_hn, t_hn = sim_engine.search_faiss_hnsw(sim_q, top_k=5)

        col1, col2, col3 = st.columns(3)
        col1.metric("NumPy Cosinus Exact", f"{t_np:.3f} ms", "Référence")
        col2.metric("FAISS IndexFlatIP", f"{t_fl:.3f} ms", f"x{t_np/max(t_fl, 1e-6):.1f} plus rapide")
        col3.metric("FAISS IndexHNSW (Graphe)", f"{t_hn:.3f} ms", f"x{t_np/max(t_hn, 1e-6):.1f} plus rapide")