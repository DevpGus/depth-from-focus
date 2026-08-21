from scipy.ndimage import convolve
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path
import pyvista as pv
import open3d as o3d
import pandas as pd
import numpy as np
import cv2 as cv
import gco
import csv
import os
import re


# ---------------------------------------------
# Funções Auxiliares
# ---------------------------------------------

# Função de Apoio - Carrega e Ordena um Conjunto de Imagens.
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
        img = cv.imread(img_path)
        
        if img is not None:
            images.append(img)
        else:
            print(f"[OS] Aviso: Não foi possível ler {filename}")

    return images

# Função de Apoio - Salva o index_map em disco no formato NumPy (.npz ou .npy).
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

# Função de Apoio - Carrega o index_map de um arquivo salvo em disco.
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

# Função de Apoio - Filtro de Sobel.
def edgeness(img):
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]) / 8
    Sy = Sx.T

    if img.ndim == 3:
        b = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0
    else:
        b = img / 255.0

    grad_x = convolve(b, Sx)
    grad_y = convolve(b, Sy)

    return np.sqrt(grad_x**2 + grad_y**2)

# Função de Apoio - Laplaciano
def laplacian(img):
    L = np.array([[1,  1, 1],
                  [1, -8, 1],
                  [1,  1, 1]])

    if img.ndim == 3:
        b = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255.0
    else:
        b = img / 255.0

    return convolve(b, L)

# Função de Apoio - Blur Gaussiano
def gaussian_blur(img):
    gauss = cv.GaussianBlur(img, (5, 5), 0)
    return gauss

# Função de Apoio - Magnitude de Nitidez
def focus(img, blur = 0):
    
    L = laplacian(img) 

    if blur:
        L = gaussian_blur(L)

    magnitude = np.absolute(L)

    return np.uint8(magnitude * 255)

# Função de Apoio - Filtro de Camada 0.
def filtrar(index_map: np.ndarray, radius: float, n: int, center: tuple[float, float] | None = None) -> np.ndarray:
    """Altera os pixels com valor 0 para -1 que estejam fora de um raio circular.

    Parameters
    ----------
    index_map : np.ndarray
        Matriz 2D contendo o mapa de índices.
    radius : float
        Raio do círculo (em pixels). O que estiver dentro do raio permanece inalterado.
    n : int
        Valor a ser atribuído aos pixels com valor 0 que estiverem fora do raio.
    center : tuple[float, float] | None, optional
        Coordenadas do centro do círculo. Se None, o centro será o centro da imagem.

    Returns
    -------
        Matriz filtrada (com dtype int32) onde pixels 0 fora do raio passam a ser n.

    """

    filter = index_map.copy()
    h, w = index_map.shape[:2]

    # Define o centro da imagem
    if center is None:
        cy, cx = h / 2.0 -100, w / 2.0 - 100 # ajuste para deslocar o centro da concha
    else:
        cy, cx = center

    # Gera grid de coordenadas (Y, X)
    y_grid, x_grid = np.ogrid[:h, :w]

    # Calcula a distância euclidiana de cada pixel em relação ao centro
    distances = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)

    # Condição: estar fora do raio E ter valor igual a 0
    mask = (distances > radius) & (filter == 0)

    # Substitui por i[-1] apenas onde a condição é verdadeira
    filter[mask] = n

    return filter

# Função de Apoio - Extraindo Metadados da Imagem
def load_metadata(metadata_path: str):
    """Carrega os metadados da câmera a partir de um arquivo CSV."""

    rows = []
    with open(metadata_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["frame"]), float(row["camera_height_mm"]),
                          float(row["focus_distance_mm"])))
    rows.sort(key=lambda r: r[0])
    camera_height_mm = np.array([r[1] for r in rows], dtype=np.float64)
    focus_distance_mm = rows[0][2]

    return camera_height_mm, focus_distance_mm


# ---------------------------------------------
# Alinhamento de Imagens (ECC + Recorte)
# ---------------------------------------------

# Função para alinhar o conjunto de imagens (ECC).
def align_images(images):
    """
    1. Alinha usando ECC.
    2. Rastreia as bordas pretas criadas pelo zoom/shift.
    3. Recorta o resultado final para eliminar áreas sem dados.
    """

    if not images: return [], []
    
    print(f"\nFase 1.2: Alinhamento ECC com Recorte ({len(images)} imagens)")
    
    # Referência - Ground Truth.
    ref_img = images[0]
    ref_gray = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY).astype(np.float32)
    height, width = ref_img.shape[:2]
    
    aligned_images = [ref_img]
    scales = [1.0]
    acummulated_scales = scales.copy()
    common_mask = np.ones((height, width), dtype=np.uint8) * 255
    
    # Configuração ECC (Critérios de parada).
    warp_mode = cv.MOTION_AFFINE
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 50, 1e-5)
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    for i in range(1, len(images)):
        img_curr = images[i]
        curr_gray = cv.cvtColor(img_curr, cv.COLOR_BGR2GRAY).astype(np.float32)
        
        try:
            # Passo ECC
            step_matrix = np.eye(2, 3, dtype=np.float32)
            cc, step_matrix = cv.findTransformECC(ref_gray, curr_gray, step_matrix, warp_mode, criteria)
            
            # Matriz Acumulada.
            step_3x3 = np.vstack([step_matrix, [0, 0, 1]])
            warp_3x3_accum = np.vstack([warp_matrix, [0,0,1]])
            combined = np.dot(step_3x3, warp_3x3_accum)
            warp_matrix = combined[:2, :]
            
            # Alinhamento da Imagem.
            img_aligned = cv.warpAffine(img_curr, warp_matrix, (width, height), 
                                         flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP,
                                         borderMode=cv.BORDER_CONSTANT, borderValue=(0,0,0))
            
            aligned_images.append(img_aligned)
            
            # Atualização da Máscara de Validade.
            mask_curr = np.ones((height, width), dtype=np.uint8) * 255            
            mask_warped = cv.warpAffine(mask_curr, warp_matrix, (width, height), 
                                         flags=cv.INTER_NEAREST + cv.WARP_INVERSE_MAP,
                                         borderMode=cv.BORDER_CONSTANT, borderValue=0)
            
            common_mask = cv.bitwise_and(common_mask, mask_warped)
            
            # Cálculo de Escalas.
            s_x = np.sqrt(warp_matrix[0,0]**2 + warp_matrix[1,0]**2)
            scales.append(s_x)
            acummulated_scales.append(acummulated_scales[-1]*s_x)
            
            print(f">> Img {i} Alinhada (ECC). Escala: {s_x:.4f}")

            # Nova Referência.
            ref_gray = curr_gray
            
        except cv.error:
            print(f"[OS] Falha ECC na img {i}. Usando anterior.")
            aligned_images.append(aligned_images[-1])
            scales.append(scales[-1])
            acummulated_scales.append(acummulated_scales[-1]**2)
            
    # Corte
    contours, _ = cv.findContours(common_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        x, y, w, h = cv.boundingRect(contours[0])
        print(f"\n[OS] Área válida detectada: \n>> X: {x}, Y: {y}, Largura: {w}, Altura: {h}")
        print(f"[OS] Pixels perdidos nas bordas: \n>> Horizontal: {width-w}, Vertical: {height-h}")
        
        final_images = []
        for img in aligned_images:
            # Recorta todas as imagens no mesmo retângulo seguro.
            cropped = img[y:y+h, x:x+w]
            final_images.append(cropped)
            
        return final_images, scales, acummulated_scales
    else:
        print("[OS] Erro Crítico: A máscara comum ficou vazia! (Overlap insuficiente). Retornando originais.")
        return aligned_images, scales, acummulated_scales

# Salva os Resultados.
def save_results(aligned_images, scales, acummulated_scales, output_folder, csv_path, resume_csv_path):
    """
    Salva as imagens processadas e um arquivo CSV com os dados de escala/zoom.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Pasta '{output_folder}' criada.")
    print(f"\n[OS] Salvando {len(aligned_images)} imagens e dados em '{output_folder}'...")
    
    with open(csv_path, mode='w', newline='') as file:
        for i, _ in enumerate(scales):
            filename = f"aligned_{i:04d}.jpg"
            filepath = os.path.join(output_folder, filename)

            cv.imwrite(filepath, aligned_images[i])

    imgs_idx = np.arange(0, len(scales), 1)
    df = pd.DataFrame(columns=['imagens', 'escalas', 'escalas_acumuladas'])
    df['imagens'] = imgs_idx
    df['escalas'] = scales
    df['escalas_acumuladas'] = acummulated_scales
    resume = df.describe()

    df.to_csv(csv_path, index=False)
    resume.to_csv(resume_csv_path, index=True)
        
    print("[OS] Concluído! Todos os arquivos foram salvos.")
    return csv_path


# ---------------------------------------------
# Mapa de Profundidade (Depth Map)
# ---------------------------------------------

# Cálculo do Mapa de Índices (Depth Map) a partir do conjunto de imagens alinhadas.
def index_map(images):
    """
    Realiza o Depth from Focus (DfF) para gerar a imagem nítida e mapas de dados.
    
    Parâmetros:
      images: Lista de imagens (numpy arrays) alinhadas.

    Retorna:
      Um dicionário com:
      - 'image': Imagem final.
      - 'depth_map': Mapa de índices.
      - 'confidence_map': Mapa de nitidez.
      - 'volume': Cubo de nitidez.
    """

    h, w = images[0].shape[:2]
    n = len(images)
    
    # 1. Alocação de Memória (Cubo de Nitidez)
    sharpness_cube = np.zeros((h, w, n), dtype=np.float32)
    
    # 2. Loop de Cálculo de Nitidez    
    for i, img in enumerate(images):
    
        ref = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        L = focus(ref, blur=0)
        
        sharpness_cube[:, :, i] = L

    # 3. Seleção de Camada
    depth_index = np.argmax(sharpness_cube, axis=2)   # Mapa de Profundidade Bruto (Índices)
    confidence_map = np.max(sharpness_cube, axis=2)   # Mapa de Confiança (Valor da Magnitude)

    # 4. Reconstrução da Imagem  
    all_in_focus = np.zeros_like(images[0])    
    rows, cols = np.indices((h, w))
    
    stack_imgs = np.stack(images, axis=-1)     
    for c in range(3): 
        all_in_focus[:, :, c] = stack_imgs[:, :, c, :][rows, cols, depth_index]
    

    return {
        'image': all_in_focus,
        'depth_map': depth_index,
        'confidence': confidence_map,
        'volume': sharpness_cube
    }

# Função para calcular a profundidade a partir do mapa de índices.
def depth(index_map, camera_height_mm, focus_distance_mm):
    """Calcula a profundidade a partir do mapa de índices.

    Parameters
    ----------
    index_map : np.ndarray
        Mapa de índices (2D).
    camera_height_mm : np.ndarray
        Altura da câmera em milímetros.
    focus_distance_mm : float
        Distância de foco em milímetros.
    passo_mm : float
        Passo entre as imagens em milímetros.

    Returns
    -------
    np.ndarray
        Mapa de profundidade (2D) em milímetros.
    """
    # Calcula a profundidade com base no índice e na altura da câmera
    depth_map = camera_height_mm[index_map] - focus_distance_mm
    return depth_map

# Cor: Pega o pixel do mapa de profundidade e retorna a cor correspondente no espectro de cores.
def color_map(images, index_map):
    """
    Para cada pixel, pega o RGB do frame mais nitido.
    """
    idx = np.rint(index_map).astype(int)
    idx = np.clip(idx, 0, images.shape[0] - 1)
    h, w = idx.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return images[idx, ys, xs, :]  # (H, W, 3) uint8


# ---------------------------------------------
# Refinamento do Mapa (Depth Map Refinement)
# ---------------------------------------------

# Função de Apoio - Graph Cut
def graph_cut(volume, weight):

    h, w, d = volume.shape

    max_focus = np.max(volume)
    data_cost = (max_focus - volume).astype(np.int32)
    data_cost = data_cost.reshape(-1, d)

    smooth_cost = (1 - np.eye(d)).astype(np.int32) * weight 

    gc = gco.GCO()
    gc.create_general_graph(h * w, d)
    gc.set_data_cost(data_cost)
    gc.set_smooth_cost(smooth_cost)

    grid_idx = np.arange(h * w).reshape(h, w)

    edges_v = np.vstack([grid_idx[:-1, :].ravel(), grid_idx[1:, :].ravel()]).T
    edges_h = np.vstack([grid_idx[:, :-1].ravel(), grid_idx[:, 1:].ravel()]).T

    all_edges = np.vstack([edges_v, edges_h]).astype(np.int32)
    weights = np.ones(all_edges.shape[0], dtype=np.int32)

    gc.set_all_neighbors(all_edges[:, 0], all_edges[:, 1], weights)
    gc.expansion()

    labels = gc.get_labels().reshape(h, w)

    return labels.astype(np.uint8)

# Função de Apoio - Filtro de Mediana Ponderada
def wmf(depth_idx, image, radius=1, sigma_s=5.0, sigma_r=0.1):

    h, w = depth_idx.shape
    refined_depth = np.zeros_like(depth_idx, dtype=np.float32)
    
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
        
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_weights = np.exp(-(x**2 + y**2) / (2 * sigma_s**2))
    
    padded_depth = np.pad(depth_idx, radius, mode='reflect')
    padded_guide = np.pad(image, radius, mode='reflect')

    for i in range(h):
        for j in range(w):

            guide_window = padded_guide[i:i+2*radius+1, j:j+2*radius+1]
            depth_window = padded_depth[i:i+2*radius+1, j:j+2*radius+1].flatten()
            
            center = guide_window[radius, radius]
            light_weights = np.exp(-(guide_window - center)**2 / (2 * sigma_r**2))
            weights = (spatial_weights * light_weights).flatten()
            
            idx = np.argsort(depth_window)
            sorted_depths = depth_window[idx]
            sorted_weights = weights[idx]

            cum_weights = np.cumsum(sorted_weights)
            median_idx = np.searchsorted(cum_weights, cum_weights[-1] / 2.0)
            
            refined_depth[i, j] = sorted_depths[median_idx]
            
    return refined_depth.astype(np.uint8)

# ----------------------------------------------
# Modelagem 3D (3D Modeling)
# ----------------------------------------------

@dataclass
class CameraParams:
    """Classe para armazenar os parâmetros da câmera."""
    focal_length_mm: float
    sensor_width_mm: float
    resolution: tuple
    sensor_height_mm: float = None

    def __post_init__(self):
        w_px, h_px = self.resolution
        if self.sensor_height_mm is None:
            self.sensor_height_mm = self.sensor_width_mm * (h_px / w_px)
        self.fx = self.focal_length_mm * w_px / self.sensor_width_mm    # px
        self.fy = self.focal_length_mm * h_px / self.sensor_height_mm  # px
        self.cx = w_px / 2.0
        self.cy = h_px / 2.0

# Função de Apoio - Calcula as coordenadas 3D (X, Y, Z) a partir do mapa de profundidade.
def pixels_to_xyz(z_fossil_mm, camera, focus_distance_mm):
    """
    Converte coordenadas de pixel para coordenadas 3D (X, Y, Z) em milímetros.

    Parameters
    ----------
    z_fossil_mm : np.ndarray
        Mapa de profundidade (2D) em milímetros.
    camera : CameraParams
        Parâmetros da câmera.
    focus_distance_mm : float
        Distância de foco em milímetros.

    Returns
    -------
    np.ndarray
        Array 3D com coordenadas (X, Y, Z) em milímetros.
    """

    h, w = z_fossil_mm.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
 
    d = focus_distance_mm  # distancia camera->ponto 
    X = (us - camera.cx) * d / camera.fx
    Y = (vs - camera.cy) * d / camera.fy
    Z = z_fossil_mm
 
    return X, Y, Z

# Função de Apoio - Gera faces (triângulos) conectando pixels vizinhos de uma grade HxW.
def build_grid(h, w, valid_mask=None):
    """Gera faces (triangulos) conectando pixels vizinhos de uma grade
    HxW. Se valid_mask for dado, pula quads com algum vertice invalido
    (ex: fundo, fora do fossil)."""
    faces = []
    for r in range(h - 1):
        for c in range(w - 1):
            if valid_mask is not None:
                quad = (valid_mask[r, c], valid_mask[r, c + 1],
                        valid_mask[r + 1, c], valid_mask[r + 1, c + 1])
                if not all(quad):
                    continue
            v00 = r * w + c
            v01 = r * w + (c + 1)
            v10 = (r + 1) * w + c
            v11 = (r + 1) * w + (c + 1)
            faces.append((v00, v10, v01))
            faces.append((v01, v10, v11))
    return faces

#  Função de Apoio - Salva um arquivo PLY (ASCII) com vértices, cores e faces.
def write_ply(path, vertices, colors, faces=None):
    """vertices: (N,3) float; colors: (N,3) uint8; faces: lista de
    tuplas de 3 indices (opcional -- se None, escreve so nuvem de pontos)."""
    n_v = len(vertices)
    n_f = len(faces) if faces is not None else 0
 
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n_v}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if faces is not None:
            f.write(f"element face {n_f}\n")
            f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
 
        for (x, y, z), (r, g, b) in zip(vertices, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
 
        if faces is not None:
            for tri in faces:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

# Função de Apoio - Visualiza o arquivo PLY em 3D usando PyVista.
def view_3d(caminho_ply: str | Path):
    caminho = Path(caminho_ply)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    # Carrega o arquivo PLY
    mesh = pv.read(caminho)

    # Configura o plotter
    plotter = pv.Plotter(window_size=[1024, 768])
    plotter.add_text(f"Fóssil: {caminho.name}", font_size=10)

    # Adiciona a geometria preservando as cores dos vértices (RGB)
    plotter.add_mesh(mesh, rgb=True, show_edges=False)
    plotter.add_axes()
    plotter.show_grid()

    # Exibe a janela
    plotter.show()

# Função de Apoio - Visualiza o arquivo PLY em 3D usando Open3D.
def view_window_3d(caminho_ply: str | Path):
    """Carrega o arquivo PLY gerado pelo pipeline e exibe na janela interativa."""
    caminho = Path(caminho_ply)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

    print(f"Carregando modelo de: {caminho}")
    mesh = o3d.io.read_triangle_mesh(str(caminho))

    if len(mesh.triangles) > 0:
        print(
            f"Exibindo Malha 3D ({len(mesh.vertices)} vértices, {len(mesh.triangles)} faces)"
        )
        # Recalcula normais.
        mesh.compute_vertex_normals()

        # Renderizar malha.
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=f"Reconstrução 3D - {caminho.name}",
            width=1024,
            height=768,
            mesh_show_back_face=True,  # Nome correto do parâmetro
        )
    else:
        # Se não tiver faces (as_mesh=False), carrega como nuvem de pontos (PointCloud).
        pcd = o3d.io.read_point_cloud(str(caminho))
        print(f"Exibindo Nuvem de Pontos ({len(pcd.points)} pontos)")

        o3d.visualization.draw_geometries(
            [pcd],
            window_name=f"Reconstrução 3D - {caminho.name}",
            width=1024,
            height=768,
        )

# Pipeline de reconstrução 3D: gera nuvem de pontos e malha colorida e grava em PLY.
def reconstruct(best_index_map, camera_height_mm, focus_distance_mm, camera,
                 frames_dir, n_frames, output_ply,
                 as_mesh=True, mask=None, stride=1):
    """Monta a nuvem de pontos / malha 3D colorida e grava em output_ply.
 
    - best_index_map: (H,W) indice (int ou float) do frame mais nitido
      por pixel, na MESMA resolucao das imagens originais (antes de
      qualquer stride).
    - mask: (H,W) bool opcional -- True = pixel pertence ao fossil.
      Sem mascara, pixels de fundo (que nunca focam de verdade) entram
      como pontos espurios/ruidosos. Recomendado gerar essa mascara no
      seu codigo de focus-measure (ex: variancia maxima da pilha acima
      de um limiar).
    - stride: subamostra a grade (1 = resolucao total).
    """
    best_index_map = np.asarray(best_index_map)
    if stride > 1:
        best_index_map = best_index_map[::stride, ::stride]
        if mask is not None:
            mask = mask[::stride, ::stride]
 
    h, w = best_index_map.shape
 
    # 1) indice -> altura real (Z)
    z_fossil_mm = depth(best_index_map, camera_height_mm,
                                      focus_distance_mm)
 
    # 2) pixel -> XY no mundo (pinhole, d = distancia de foco fixa)
    X, Y, Z = pixels_to_xyz(z_fossil_mm, camera, focus_distance_mm)
 
    # 3) cor = pixel do frame mais nitido
    images = load_images(frames_dir)
    image_stack = np.stack(images, axis=0)
    colors = color_map(image_stack, best_index_map)
 
    vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
 
    if mask is None:
        faces = build_grid(h, w) if as_mesh else None
        write_ply(output_ply, vertices, colors_flat, faces)
    else:
        # com mascara: filtra pontos invalidos da nuvem, e (se malha)
        # gera faces so entre vizinhos validos e depois remapeia indices
        mask_flat = mask.reshape(-1)
        if as_mesh:
            faces = build_grid(h, w, valid_mask=mask)
            used = sorted(set(i for tri in faces for i in tri))
            remap = {old: new for new, old in enumerate(used)}
            faces = [(remap[a], remap[b], remap[c]) for a, b, c in faces]
            write_ply(output_ply, vertices[used], colors_flat[used], faces)
        else:
            write_ply(output_ply, vertices[mask_flat], colors_flat[mask_flat], None)
 
    print(f"[reconstruct] gravado em {output_ply} "
          f"({h}x{w} pixels, {'malha' if as_mesh else 'nuvem de pontos'})")

