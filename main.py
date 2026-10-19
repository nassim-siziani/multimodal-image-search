import os
import torch
import faiss
import numpy as np
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel

# 1. Préparation du dossier et d'images d'exemple
IMAGE_DIR = "data/images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def generate_sample_images():
    """Crée 4 images simples de démonstration si le dossier est vide."""
    samples = {
        "chat_noir.jpg": ("black", "Chat noir"),
        "plage_ensoleillee.jpg": ("gold", "Plage de sable"),
        "foret_verte.jpg": ("forestgreen", "Arbres et foret"),
        "voiture_rouge.jpg": ("firebrick", "Voiture rouge de sport")
    }
    for filename, (color, label) in samples.items():
        path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(path):
            img = Image.new("RGB", (300, 300), color=color)
            draw = ImageDraw.Draw(img)
            draw.text((30, 140), label, fill="white")
            img.save(path)

# 2. Chargement du modèle CLIP
print(">>> Chargement du modèle CLIP (openai/clip-vit-base-patch32)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

def extract_tensor(features):
    """Garantit l'extraction d'un tenseur PyTorch quelle que soit la version de transformers."""
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "last_hidden_state"):
        return features.last_hidden_state[:, 0, :]
    return features

# 3. Extraction et indexation des images
def build_index(image_folder):
    generate_sample_images()
    image_paths = [
        os.path.join(image_folder, f) 
        for f in os.listdir(image_folder) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

    if not image_paths:
        raise RuntimeError("Aucune image trouvée dans le dossier data/images.")

    print(f">>> Encodage de {len(image_paths)} images...")
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = extract_tensor(image_features)
        # Normalisation L2 pour la similarité cosinus
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings = image_features.cpu().numpy().astype("float32")

    # Index FAISS basé sur le produit scalaire (Inner Product)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, image_paths

# 4. Fonction de recherche textuelle
def search_text_to_image(query, index, image_paths, top_k=2):
    print(f"\n--- Recherche pour la requête : \"{query}\" ---")
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = extract_tensor(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        query_vector = text_features.cpu().numpy().astype("float32")

    scores, indices = index.search(query_vector, top_k)

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        filename = os.path.basename(image_paths[idx])
        print(f"[{rank}] Score: {score:.4f} | Fichier: {filename}")

if __name__ == "__main__":
    faiss_index, paths = build_index(IMAGE_DIR)
    
    # Exemples de tests text-to-image
    queries = [
        "a black cat resting",
        "a sunny beach with yellow sand",
        "a red fast sports car",
        "nature and green trees"
    ]
    for q in queries:
        search_text_to_image(q, faiss_index, paths, top_k=2)
