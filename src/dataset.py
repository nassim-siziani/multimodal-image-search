"""Gestion de la base d'images : chargement d'un dossier local ou génération d'un jeu de démonstration."""

from pathlib import Path
from PIL import Image
from torchvision.datasets import CIFAR10


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def prepare_demo_dataset(output_dir: str = "data/demo_images", num_images: int = 200) -> tuple[list[str], list[str]]:
    """
    Télécharge et sauvegarde un sous-ensemble d'images réelles (CIFAR-10)
    pour permettre de tester immédiatement le moteur de recherche et l'évaluation.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    existing_files = sorted(out_path.glob("*.png"))
    if len(existing_files) >= num_images:
        image_paths = [str(p) for p in existing_files[:num_images]]
        labels = [p.stem.split("_")[0] for p in existing_files[:num_images]]
        return image_paths, labels

    print(f"Préparation de {num_images} images de démonstration dans '{output_dir}'...")
    dataset = CIFAR10(root="data/raw_cifar", train=False, download=True)

    image_paths = []
    labels = []

    for idx in range(min(num_images, len(dataset))):
        img, class_idx = dataset[idx]
        label_name = CIFAR10_CLASSES[class_idx]
        # Redimensionnement en 224x224 pour un affichage net dans l'interface
        img_resized = img.resize((224, 224), Image.Resampling.BICUBIC)
        file_path = out_path / f"{label_name}_{idx:04d}.png"
        img_resized.save(file_path)
        image_paths.append(str(file_path))
        labels.append(label_name)

    return image_paths, labels


def load_custom_directory(directory: str) -> list[str]:
    """Charge toutes les images (.jpg, .jpeg, .png, .webp) présentes dans un dossier local."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        str(p) for p in dir_path.rglob("*") if p.suffix.lower() in extensions
    )