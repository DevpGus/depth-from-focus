"""
core/pipeline.py

Funcoes de processamento do pipeline de reconstrucao 3D a partir de uma pilha
focal (imagens capturadas ao longo do trilho vertical do microscopio).

Estas sao as implementacoes REAIS fornecidas pelo usuario, adaptadas para:
  - operar em memoria (bytes/arrays) em vez de ler/escrever pastas no disco,
    ja que a UI recebe uploads;
  - usar `@st.cache_data`, para nao reprocessar a pilha inteira a cada
    interacao com um controle de uma etapa POSTERIOR do pipeline;
  - devolver dicionarios com chaves estaveis, consumidas pelas telas em
    `steps/`.

Fora do escopo desta conversao (por serem lentas): `graph_cut` e `wmf`.
Mantidas mas NAO usadas pela interface (bloqueariam o servidor por abrirem
janelas nativas): `view_3d` (PyVista) e `view_window_3d` (Open3D).

Convencao de cor: como no codigo original, as imagens trafegam em BGR (padrao
OpenCV) do carregamento ate a composicao "all-in-focus". A conversao para RGB
acontece so na hora de colorir a nuvem/malha 3D (Plotly) e no export.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import streamlit as st
from scipy.ndimage import convolve

try:
    import pyvista as pv
except ImportError:  # opcional, so necessario para view_3d()
    pv = None

try:
    import open3d as o3d
except ImportError:  # opcional, so necessario para view_window_3d()
    o3d = None


# ---------------------------------------------------------------------------
# Funcoes auxiliares
# ---------------------------------------------------------------------------

def _extract_number(filename: str) -> int:
    """Extrai o primeiro numero embutido no nome do arquivo, para ordenacao."""
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def edgeness(img: np.ndarray) -> np.ndarray:
    """Filtro de Sobel — magnitude de gradiente normalizada (0-1).

    Metrica de nitidez alternativa, disponivel para extensao futura (nao
    usada na etapa de profundidade por padrao — o padrao e `focus`, baseado
    no Laplaciano).
    """
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    Sy = Sx.T
    b = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0 if img.ndim == 3 else img / 255.0
    grad_x = convolve(b, Sx)
    grad_y = convolve(b, Sy)
    return np.sqrt(grad_x ** 2 + grad_y ** 2)


def laplacian(img: np.ndarray) -> np.ndarray:
    """Filtro Laplaciano 3x3 (kernel de 8 vizinhos), normalizado (0-1)."""
    L = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    b = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0 if img.ndim == 3 else img / 255.0
    return convolve(b, L)


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Blur gaussiano 5x5."""
    return cv.GaussianBlur(img, (5, 5), 0)


def focus(img: np.ndarray, blur: bool = False) -> np.ndarray:
    """Magnitude de nitidez (uint8, 0-255) a partir do Laplaciano.

    Input: imagem (grayscale ou BGR), flag de suavizacao previa.
    Descricao: base do shape-from-focus — quanto maior o valor, mais nitido
        o pixel esta naquela fatia.
    Output: mapa de nitidez uint8, mesma resolucao da entrada.
    """
    L = laplacian(img)
    if blur:
        L = gaussian_blur(L)
    return np.uint8(np.absolute(L) * 255)


def filtrar(
    index_map_arr: np.ndarray, radius: float, n: int, center: tuple[float, float] | None = None
) -> np.ndarray:
    """Marca com `n` os pixels de fundo (valor 0) que estao fora de um raio
    circular. Util para descartar bordas/fundo de amostras circulares.
    Disponivel para extensao futura (nao chamada pela UI por padrao).
    """
    filtered = index_map_arr.copy()
    h, w = index_map_arr.shape[:2]
    if center is None:
        cy, cx = h / 2.0 - 100, w / 2.0 - 100
    else:
        cy, cx = center
    y_grid, x_grid = np.ogrid[:h, :w]
    distances = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
    mask = (distances > radius) & (filtered == 0)
    filtered[mask] = n
    return filtered


def load_metadata(metadata_bytes: bytes) -> tuple[np.ndarray, float]:
    """
    Input: conteudo binario de um CSV com colunas frame, camera_height_mm,
        focus_distance_mm.
    Descricao: le e ordena os metadados de captura por numero de frame.
    Output: (camera_height_mm: np.ndarray, focus_distance_mm: float)
    """
    text = metadata_bytes.decode("utf-8")
    rows = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        rows.append(
            (int(row["frame"]), float(row["camera_height_mm"]), float(row["focus_distance_mm"]))
        )
    rows.sort(key=lambda r: r[0])
    camera_height_mm = np.array([r[1] for r in rows], dtype=np.float64)
    focus_distance_mm = rows[0][2]
    return camera_height_mm, focus_distance_mm


# ---------------------------------------------------------------------------
# Utilitarios opcionais de disco (uso fora do dashboard, ex.: scripts/CLI)
# ---------------------------------------------------------------------------

def save_index_map(index_map_arr: np.ndarray, caminho_arquivo: str | Path, comprimir: bool = True) -> None:
    """Salva uma matriz de index_map em disco (.npz ou .npy)."""
    caminho = Path(caminho_arquivo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    if comprimir:
        if caminho.suffix != ".npz":
            caminho = caminho.with_suffix(".npz")
        np.savez_compressed(caminho, index_map=index_map_arr)
    else:
        if caminho.suffix != ".npy":
            caminho = caminho.with_suffix(".npy")
        np.save(caminho, index_map_arr)


def load_index_map(caminho_arquivo: str | Path) -> np.ndarray:
    """Carrega um index_map salvo previamente em disco (.npy ou .npz)."""
    caminho = Path(caminho_arquivo)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {caminho}")
    if caminho.suffix == ".npz":
        with np.load(caminho) as dados:
            return dados["index_map"]
    return np.load(caminho)


def save_results(aligned_images, scales, acummulated_scales, output_folder, csv_path, resume_csv_path) -> str:
    """Salva as imagens alinhadas em disco e um CSV com os dados de escala/zoom.
    Util para depuracao fora do dashboard; a UI usa os resultados direto da
    memoria via `align_stack`.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(aligned_images):
        cv.imwrite(str(Path(output_folder) / f"aligned_{i:04d}.jpg"), img)

    df = pd.DataFrame(
        {
            "imagens": np.arange(len(scales)),
            "escalas": scales,
            "escalas_acumuladas": acummulated_scales,
        }
    )
    df.to_csv(csv_path, index=False)
    df.describe().to_csv(resume_csv_path, index=True)
    return csv_path


# ---------------------------------------------------------------------------
# Etapa 1 - Carregamento e ordenacao da pilha
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_image_stack(file_bytes: tuple[bytes, ...], filenames: tuple[str, ...]) -> dict:
    """
    Input:
        file_bytes: bytes de cada imagem enviada pelo usuario.
        filenames: nomes originais dos arquivos (mesma ordem de file_bytes).

    Descricao:
        Ordena a pilha pelo numero embutido em cada nome de arquivo (mesma
        logica de `extract_number`/`load_images` do pipeline original),
        decodifica cada imagem em BGR (`cv.imdecode`) e padroniza as
        dimensoes pela menor imagem da pilha.

    Output:
        dict {
            "images": list[np.ndarray],  # BGR uint8, HxWx3
            "filenames": list[str],
            "shape": tuple[int, int],
            "n_slices": int,
        }
    """
    order = sorted(range(len(filenames)), key=lambda i: _extract_number(filenames[i]))
    sorted_bytes = [file_bytes[i] for i in order]
    sorted_names = [filenames[i] for i in order]

    images = []
    for b in sorted_bytes:
        arr = np.frombuffer(b, dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    min_h = min(im.shape[0] for im in images)
    min_w = min(im.shape[1] for im in images)
    normalized = [
        cv.resize(im, (min_w, min_h)) if im.shape[:2] != (min_h, min_w) else im for im in images
    ]

    return {
        "images": normalized,
        "filenames": sorted_names,
        "shape": (min_h, min_w),
        "n_slices": len(normalized),
    }


# ---------------------------------------------------------------------------
# Etapa 2 - Alinhamento (ECC sequencial + recorte da regiao comum)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def align_stack(images: list[np.ndarray]) -> dict:
    """
    Input:
        images: pilha bruta, ja ordenada (BGR uint8).

    Descricao:
        Alinhamento sequencial via ECC (`cv.findTransformECC`, modelo afim):
        cada fatia e alinhada contra a fatia ANTERIOR, com a transformacao
        acumulada desde a primeira fatia. Ao final, recorta toda a pilha
        para a regiao comum valida (elimina bordas pretas geradas pelo
        warp) — mesma logica de `align_images` do pipeline original.

    Output:
        dict {
            "aligned_images": list[np.ndarray],
            "scales": list[float],             # fator de zoom por fatia
            "acummulated_scales": list[float],
        }
    """
    if not images:
        return {"aligned_images": [], "scales": [], "acummulated_scales": []}

    ref_img = images[0]
    ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY).astype(np.float32)
    height, width = ref_img.shape[:2]

    aligned_images = [ref_img]
    scales = [1.0]
    acummulated_scales = scales.copy()
    common_mask = np.ones((height, width), dtype=np.uint8) * 255

    warp_mode = cv.MOTION_AFFINE
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 1e-5)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    for i in range(1, len(images)):
        curr_gray = cv.cvtColor(images[i], cv.COLOR_BGR2GRAY).astype(np.float32)
        try:
            step_matrix = np.eye(2, 3, dtype=np.float32)
            _, step_matrix = cv.findTransformECC(ref_gray, curr_gray, step_matrix, warp_mode, criteria)

            step_3x3 = np.vstack([step_matrix, [0, 0, 1]])
            warp_3x3_accum = np.vstack([warp_matrix, [0, 0, 1]])
            warp_matrix = (step_3x3 @ warp_3x3_accum)[:2, :]

            img_aligned = cv.warpAffine(
                images[i], warp_matrix, (width, height),
                flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
                borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            aligned_images.append(img_aligned)

            mask_curr = np.ones((height, width), dtype=np.uint8) * 255
            mask_warped = cv.warpAffine(
                mask_curr, warp_matrix, (width, height),
                flags=cv.INTER_NEAREST + cv.WARP_INVERSE_MAP,
                borderMode=cv.BORDER_CONSTANT, borderValue=0,
            )
            common_mask = cv.bitwise_and(common_mask, mask_warped)

            s_x = np.sqrt(warp_matrix[0, 0] ** 2 + warp_matrix[1, 0] ** 2)
            scales.append(float(s_x))
            acummulated_scales.append(float(acummulated_scales[-1] * s_x))

            ref_gray = curr_gray
        except cv.error:
            aligned_images.append(aligned_images[-1])
            scales.append(scales[-1])
            acummulated_scales.append(acummulated_scales[-1] ** 2)

    contours, _ = cv.findContours(common_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv.boundingRect(contours[0])
        final_images = [img[y : y + h, x : x + w] for img in aligned_images]
    else:
        final_images = aligned_images

    return {
        "aligned_images": final_images,
        "scales": scales,
        "acummulated_scales": acummulated_scales,
    }


# ---------------------------------------------------------------------------
# Etapa 3 - Mapa de profundidade (shape-from-focus)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def estimate_depth_map(aligned_images: list[np.ndarray], blur: bool) -> dict:
    """
    Input:
        aligned_images: pilha alinhada (BGR uint8).
        blur: se True, suaviza o mapa de nitidez de cada fatia (gaussian 5x5)
            antes de comparar entre fatias — equivalente ao parametro `blur`
            de `focus()`.

    Descricao:
        Para cada fatia, calcula a magnitude de nitidez (`focus`, baseado no
        Laplaciano). Para cada pixel, seleciona o indice de fatia com maior
        nitidez (`index_map` do pipeline original) e reconstroi a imagem
        composta "all-in-focus" pegando, em cada posicao, o pixel da fatia
        mais nitida.

    Output:
        dict {
            "depth_map": np.ndarray HxW int32,        # indice bruto de fatia
            "confidence_map": np.ndarray HxW float32, # magnitude de nitidez
            "composite_image": np.ndarray HxWx3 uint8, # BGR
            "n_slices": int,
        }
    """
    h, w = aligned_images[0].shape[:2]
    n = len(aligned_images)
    sharpness_cube = np.zeros((h, w, n), dtype=np.float32)

    for i, img in enumerate(aligned_images):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sharpness_cube[:, :, i] = focus(gray, blur=blur)

    depth_index = np.argmax(sharpness_cube, axis=2)
    confidence_map = np.max(sharpness_cube, axis=2)

    stack_imgs = np.stack(aligned_images, axis=-1)
    rows, cols = np.indices((h, w))
    all_in_focus = np.zeros_like(aligned_images[0])
    for c in range(3):
        all_in_focus[:, :, c] = stack_imgs[:, :, c, :][rows, cols, depth_index]

    return {
        "depth_map": depth_index.astype(np.int32),
        "confidence_map": confidence_map.astype(np.float32),
        "composite_image": all_in_focus,
        "n_slices": n,
    }


# ---------------------------------------------------------------------------
# Etapa 4 - Reconstrucao 3D (camera pinhole + triangulacao)
# ---------------------------------------------------------------------------

@dataclass
class CameraParams:
    """Parametros de camera (modelo pinhole)."""

    focal_length_mm: float
    sensor_width_mm: float
    resolution: tuple
    sensor_height_mm: float = None

    def __post_init__(self):
        w_px, h_px = self.resolution
        if self.sensor_height_mm is None:
            self.sensor_height_mm = self.sensor_width_mm * (h_px / w_px)
        self.fx = self.focal_length_mm * w_px / self.sensor_width_mm
        self.fy = self.focal_length_mm * h_px / self.sensor_height_mm
        self.cx = w_px / 2.0
        self.cy = h_px / 2.0


def _depth_from_index(index_map_arr: np.ndarray, camera_height_mm: np.ndarray, focus_distance_mm: float) -> np.ndarray:
    """Converte indice de fatia em profundidade fisica (mm) — `depth()` original."""
    idx = np.clip(index_map_arr, 0, len(camera_height_mm) - 1)
    return camera_height_mm[idx] - focus_distance_mm


def pixels_to_xyz(z_mm: np.ndarray, camera: CameraParams, focus_distance_mm: float):
    """Converte grade de pixels + profundidade (mm) em coordenadas 3D (X, Y, Z) mm."""
    h, w = z_mm.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    d = focus_distance_mm
    X = (us - camera.cx) * d / camera.fx
    Y = (vs - camera.cy) * d / camera.fy
    Z = z_mm
    return X, Y, Z


def build_grid(h: int, w: int, valid_mask: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Gera faces (triangulos) conectando pixels vizinhos de uma grade HxW.
    Se `valid_mask` for dado, pula quads com algum vertice invalido (fundo).
    """
    faces = []
    for r in range(h - 1):
        for c in range(w - 1):
            if valid_mask is not None:
                quad = (
                    valid_mask[r, c], valid_mask[r, c + 1],
                    valid_mask[r + 1, c], valid_mask[r + 1, c + 1],
                )
                if not all(quad):
                    continue
            v00 = r * w + c
            v01 = r * w + (c + 1)
            v10 = (r + 1) * w + c
            v11 = (r + 1) * w + (c + 1)
            faces.append((v00, v10, v01))
            faces.append((v01, v10, v11))
    return faces


@st.cache_data(show_spinner=False)
def reconstruct_3d(
    depth_map: np.ndarray,
    composite_image: np.ndarray,
    confidence_map: np.ndarray,
    camera_height_mm: np.ndarray,
    focus_distance_mm: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    downsample: int,
    confidence_threshold: float,
    as_mesh: bool,
) -> dict:
    """
    Input:
        depth_map: indice de fatia por pixel (HxW int).
        composite_image: imagem all-in-focus BGR (fonte de cor).
        confidence_map: magnitude de nitidez por pixel (usada para mascarar
            fundo/pontos espurios).
        camera_height_mm: posicao (mm) de cada fatia ao longo do trilho.
        focus_distance_mm: distancia de foco fixa (mm).
        focal_length_mm, sensor_width_mm: parametros da lente/sensor
            (modelo pinhole).
        downsample: subamostragem da grade antes da triangulacao.
        confidence_threshold: percentil (0-1) de confianca minima; pixels
            abaixo sao tratados como fundo e removidos. 0 desativa.
        as_mesh: se True, gera faces (triangulos); se False, so nuvem de
            pontos.

    Descricao:
        Reimplementa `reconstruct()` do pipeline original operando em
        memoria: indice -> profundidade fisica (`depth`) -> coordenadas 3D
        pinhole (`pixels_to_xyz`) -> cor (pixel mais nitido) -> triangulacao
        da grade valida (`build_grid`).

    Output:
        dict {
            "vertices": np.ndarray Nx3 float32,   # (x, y, z) em mm
            "colors": np.ndarray Nx3 uint8,       # RGB
            "faces": list[tuple[int,int,int]] | None,
            "grid_shape": tuple[int, int],
        }
    """
    dm = depth_map[::downsample, ::downsample]
    cm = confidence_map[::downsample, ::downsample]
    img = composite_image[::downsample, ::downsample]
    h, w = dm.shape

    z_mm = _depth_from_index(dm, camera_height_mm, focus_distance_mm)
    resolution = (composite_image.shape[1], composite_image.shape[0])
    camera = CameraParams(focal_length_mm=focal_length_mm, sensor_width_mm=sensor_width_mm, resolution=resolution)
    X, Y, Z = pixels_to_xyz(z_mm, camera, focus_distance_mm)

    colors_bgr = img.reshape(-1, 3)
    colors_rgb = colors_bgr[:, ::-1]
    vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    mask = None
    if confidence_threshold > 0:
        thresh = np.percentile(cm, confidence_threshold * 100)
        mask = cm >= thresh

    if mask is None:
        faces = build_grid(h, w) if as_mesh else None
        used_vertices, used_colors = vertices, colors_rgb
    else:
        mask_flat = mask.reshape(-1)
        if as_mesh:
            faces_raw = build_grid(h, w, valid_mask=mask)
            used_idx = sorted(set(i for tri in faces_raw for i in tri))
            remap = {old: new for new, old in enumerate(used_idx)}
            faces = [(remap[a], remap[b], remap[c]) for a, b, c in faces_raw]
            used_vertices = vertices[used_idx]
            used_colors = colors_rgb[used_idx]
        else:
            faces = None
            used_vertices = vertices[mask_flat]
            used_colors = colors_rgb[mask_flat]

    return {
        "vertices": used_vertices,
        "colors": used_colors.astype(np.uint8),
        "faces": faces,
        "grid_shape": (h, w),
    }


# ---------------------------------------------------------------------------
# Etapa 5 - Exportacao
# ---------------------------------------------------------------------------

def export_ply(mesh_result: dict) -> bytes:
    """Serializa vertices/cores/faces em PLY ASCII em memoria — adaptacao de
    `write_ply` do pipeline original (escreve em bytes, nao em disco).
    """
    vertices = mesh_result["vertices"]
    colors = mesh_result["colors"]
    faces = mesh_result.get("faces")

    buf = io.StringIO()
    buf.write("ply\nformat ascii 1.0\n")
    buf.write(f"element vertex {len(vertices)}\n")
    buf.write("property float x\nproperty float y\nproperty float z\n")
    buf.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
    if faces:
        buf.write(f"element face {len(faces)}\n")
        buf.write("property list uchar int vertex_indices\n")
    buf.write("end_header\n")
    for (x, y, z), (r, g, b) in zip(vertices, colors):
        buf.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
    if faces:
        for a, b_, c in faces:
            buf.write(f"3 {a} {b_} {c}\n")
    return buf.getvalue().encode("utf-8")


def export_xyz(mesh_result: dict) -> bytes:
    """Serializa apenas a nuvem de pontos (sem faces) em formato XYZ (texto)."""
    vertices = mesh_result["vertices"]
    colors = mesh_result["colors"]
    buf = io.StringIO()
    for (x, y, z), (r, g, b) in zip(vertices, colors):
        buf.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
    return buf.getvalue().encode("utf-8")


def export_point_cloud(mesh_result: dict, fmt: str) -> bytes:
    """Dispatcher usado pela UI. fmt em {'PLY', 'XYZ'}."""
    return export_ply(mesh_result) if fmt == "PLY" else export_xyz(mesh_result)