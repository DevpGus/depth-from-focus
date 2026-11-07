"""
core/synthetic.py

Gera uma pilha focal sintetica (nao vem de arquivo nenhum) para o modo
"Usar pilha de exemplo" do dashboard, permitindo testar o fluxo completo sem
precisar de imagens reais do microscopio.
"""

from __future__ import annotations

import numpy as np


def generate_synthetic_stack(n_slices: int = 12, size: int = 256, seed: int = 7) -> list[np.ndarray]:
    """
    Input:
        n_slices: numero de fatias da pilha sintetica.
        size: dimensao (size x size) de cada fatia.
        seed: semente aleatoria para reprodutibilidade.

    Descricao:
        Cria uma superficie 3D sintetica (soma de gaussianas simulando
        "crateras" de uma amostra) e, para cada fatia z, desfoca com
        intensidade proporcional a distancia entre z e a profundidade real de
        cada ponto - simulando uma pilha focal real, em que cada imagem tem
        uma regiao nitida e o resto desfocado.

    Output:
        list[np.ndarray] com n_slices imagens HxWx3 uint8.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]

    depth = np.zeros((size, size))
    for _ in range(5):
        cx, cy = rng.uniform(0.15, 0.85, 2) * size
        sigma = rng.uniform(size * 0.08, size * 0.18)
        amp = rng.uniform(0.4, 1.0)
        depth += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    texture = rng.normal(0.5, 0.12, (size, size))
    texture = np.clip(texture, 0, 1)

    from scipy import ndimage

    stack = []
    for i in range(n_slices):
        z_plane = i / max(n_slices - 1, 1)
        blur_amount = np.abs(depth - z_plane) * 6.0 + 0.3
        # Desfoque espacialmente variavel aproximado: aplica alguns niveis de
        # blur e mistura por regiao de acordo com blur_amount.
        levels = [0.3, 1.0, 2.0, 3.5, 5.0, 7.0]
        blurred_versions = [ndimage.gaussian_filter(texture, sigma=s) for s in levels]
        idx = np.clip((blur_amount / 7.0) * (len(levels) - 1), 0, len(levels) - 1).astype(int)
        plane = np.choose(idx, blurred_versions)

        gray = np.clip(plane * 255, 0, 255).astype(np.uint8)
        rgb = np.stack([gray, (gray * 0.95).astype(np.uint8), (gray * 1.05).clip(0, 255).astype(np.uint8)], axis=-1)
        stack.append(rgb)

    return stack