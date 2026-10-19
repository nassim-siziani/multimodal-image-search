"""Extraction et normalisation L2 des représentations vectorielles (embeddings) avec CLIP."""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """
    Normalise chaque vecteur ligne selon la norme euclidienne L2.
    Une fois normalisés, le produit scalaire entre deux vecteurs est égal à leur similarité cosinus.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return (vectors / norms).astype(np.float32)


class CLIPEmbedder:
    """Interface autour du modèle multimodal CLIP (Contrastive Language-Image Pre-Training)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Transforme une liste de descriptions textuelles en vecteurs normalisés L2 (N, 512)."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        if not isinstance(outputs, torch.Tensor):
            outputs = outputs.pooler_output
        vectors = outputs.cpu().numpy()
        return l2_normalize(vectors)

    @torch.no_grad()
    def embed_images(self, image_paths_or_pil: list[str | Image.Image], batch_size: int = 32) -> np.ndarray:
        """Transforme une liste d'images en vecteurs normalisés L2 (N, 512) par lots (batches)."""
        all_vectors = []

        for i in range(0, len(image_paths_or_pil), batch_size):
            batch = image_paths_or_pil[i : i + batch_size]
            pil_images = [
                Image.open(item).convert("RGB") if isinstance(item, str) else item.convert("RGB")
                for item in batch
            ]
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            if not isinstance(outputs, torch.Tensor):
                outputs = outputs.pooler_output
            all_vectors.append(outputs.cpu().numpy())

        vectors = np.vstack(all_vectors)
        return l2_normalize(vectors)