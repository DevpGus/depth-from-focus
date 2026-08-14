import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import cv2
import os
import re

# Carrega e Ordena um Conjunto de Imagens.
def load_images(folder):
    """
    Ordena um conjunto de imagens a partir de uma dataset contendo ordenação 
    númerica crescente a patir da localização da pasta de origem.
    
    """
    
    images = []
    
    try:
        all_files = os.listdir(folder)
    except FileNotFoundError:
        print(f"[OS] Erro: A pasta '{folder}' não foi encontrada.")
        return []

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [f for f in all_files if f.lower().endswith(valid_extensions)]

    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0 

    sorted_files = sorted(image_files, key=extract_number)
    
    print(f"\nFase 1.1: Ordenação.\nOrdem de carregamento detectada: {sorted_files[:3]} ...")

    for filename in sorted_files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            images.append(img)
        else:
            print(f"[OS] Aviso: Não foi possível ler {filename}")

    return images

# Salva o index_map em disco no formato NumPy (.npz ou .npy).
def save_index_map(
    index_map: np.ndarray, caminho_arquivo: str | Path, comprimir: bool = True
) -> None:
    """Salva uma matriz de index_map em disco no formato NumPy (.npz ou .npy).

    Parâmetros
    ----------
    index_map : np.ndarray
        Matriz/array contendo o mapa de índices.
    caminho_arquivo : str | Path
        Caminho de destino do arquivo.
    comprimir : bool, optional
        Se True, salva comprimido (.npz). Se False, salva não comprimido (.npy).

    """
    caminho = Path(caminho_arquivo)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    if comprimir:
        if caminho.suffix != ".npz":
            caminho = caminho.with_suffix(".npz")
        np.savez_compressed(caminho, index_map=index_map)
    else:
        if caminho.suffix != ".npy":
            caminho = caminho.with_suffix(".npy")
        np.save(caminho, index_map)

# Carrega o index_map de um arquivo salvo em disco.
def load_index_map(caminho_arquivo: str | Path) -> np.ndarray:
    """Extrai/carrega uma matriz de index_map salva previamente em disco.

    Parâmetros
    ----------
    caminho_arquivo : str | Path
        Caminho do arquivo (.npy ou .npz).

    Returns
    -------
    np.ndarray
        Matriz do index_map reconstruída.

    """
    caminho = Path(caminho_arquivo)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    if caminho.suffix == ".npz":
        with np.load(caminho) as dados:
            return dados["index_map"]
    else:
        return np.load(caminho)