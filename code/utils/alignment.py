import matplotlib.pyplot as plt
import pandas as pd
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