# Moteur de recherche d'images multimodal (CLIP & FAISS)

## Présentation et but du projet

Les moteurs de recherche d'images classiques reposent souvent sur les noms de fichiers ou sur des mots-clés ajoutés manuellement. Si une image n'a pas de légende, il devient très difficile de la retrouver à partir d'une description écrite.

**Le but de ce projet** est de créer un moteur de recherche sémantique capable de comprendre à la fois le texte et le contenu visuel des images. Grâce au modèle **CLIP** (développé par OpenAI), les images et les phrases sont transformées en vecteurs numériques dans un même espace mathématique. 

L'objectif est de permettre à l'utilisateur de taper une description en langage naturel (par exemple : *"un chien qui court sur la plage"*) et de retrouver instantanément les images les plus proches grâce à une recherche vectorielle rapide avec **FAISS**.

## Travail réalisé

Dans ce projet, j'ai travaillé sur plusieurs étapes :

- **Extraction des caractéristiques (Embeddings) :** utilisation du modèle pré-entraîné CLIP (`ViT-B/32`) via Hugging Face et PyTorch pour convertir chaque image et chaque requête texte en un vecteur de dimension 512.

- **Normalisation et similarité cosinus :** normalisation des vecteurs (norme L2) afin de mesurer directement la proximité sémantique entre une phrase et une image par produit scalaire.

- **Indexation vectorielle avec FAISS :** création d'un index `IndexFlatIP` permettant de stocker les vecteurs d'images et d'effectuer des recherches des $k$ plus proches voisins (*k-NN*) en quelques millisecondes.

- **Pipeline de recherche :** développement des fonctions permettant d'indexer un dossier d'images, de rechercher à partir d'une requête texte (*Text-to-Image*) ou à partir d'une autre image (*Image-to-Image*).

- **Tests unitaires :** vérification de la dimension des vecteurs, de la normalisation et de la précision de la recherche avec `pytest`.

## Technologies utilisées

- Python
- PyTorch
- Transformers (Hugging Face - CLIP)
- FAISS (`faiss-cpu`)
- Pillow (PIL)
- NumPy
- Pytest

## Lancer le projet

Clonez le dépôt :

    git clone git@github.com:nassim-siziani/multimodal-image-search.git
    cd multimodal-image-search

Créez l'environnement virtuel et installez les dépendances :

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Lancez les tests unitaires :

    python3 -m pytest tests/

Lancez une démonstration de recherche sémantique :

    python3 -m src.search_engine

## Ce que j'ai appris

Ce projet m'a permis de comprendre le fonctionnement des modèles multimodaux (Vision-Language Models) et la manière dont on peut relier du texte et des images dans un espace vectoriel commun.

Il m'a également permis de manipuler une base de données vectorielle avec FAISS pour accélérer la recherche de similarité et de structurer un projet d'intelligence artificielle propre et testé.
