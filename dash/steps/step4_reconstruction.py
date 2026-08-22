from __future__ import annotations

import numpy as np
import streamlit as st

from core.pipeline import load_metadata, reconstruct_3d
from core.state import unlock_next
from ui.nav import render_step_header


def render() -> None:
    render_step_header(
        "Reconstrucao 3D",
        "Z03",
        "Converte o mapa de profundidade em coordenadas fisicas (mm), via modelo de "
        "camera pinhole, e triangula a superficie.",
    )

    depth_result = st.session_state.depth_result
    if depth_result is None:
        st.warning("Volte a etapa Z02 e calcule o mapa de profundidade.")
        return

    n_slices = depth_result["n_slices"]

    st.markdown('<div class="console-eyebrow">Metadados do trilho</div>', unsafe_allow_html=True)
    metadata_file = st.file_uploader(
        "CSV com colunas frame, camera_height_mm, focus_distance_mm — opcional",
        type=["csv"],
    )

    if metadata_file is not None:
        camera_height_mm, focus_distance_mm = load_metadata(metadata_file.getvalue())
        st.caption(f"Metadados carregados: {len(camera_height_mm)} posicoes do trilho.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            step_mm = st.number_input(
                "Passo entre fatias no trilho (mm)", 0.001, 10.0, 0.05, step=0.005, format="%.3f"
            )
        with c2:
            focus_distance_mm = st.number_input("Distancia de foco fixa (mm)", 1.0, 500.0, 50.0, step=1.0)
        camera_height_mm = np.arange(n_slices) * step_mm

    st.markdown('<div class="console-eyebrow">Camera (modelo pinhole)</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="console-panel">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            focal_length_mm = st.number_input("Distancia focal (mm)", 1.0, 200.0, 25.0, step=1.0)
        with c2:
            sensor_width_mm = st.number_input("Largura do sensor (mm)", 1.0, 100.0, 23.5, step=0.5)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="console-eyebrow">Malha</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="console-panel">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            downsample = st.slider(
                "Subamostragem da grade", 1, 10, 4,
                help="Valores maiores = menos pontos = triangulacao mais rapida.",
            )
        with c2:
            confidence_threshold = st.slider(
                "Filtrar fundo (percentil de confianca minima)", 0.0, 0.95, 0.0, step=0.05,
                help="0 desativa. Remove pixels com nitidez de pico baixa (fundo/ruido).",
            )
        with c3:
            as_mesh = st.toggle(
                "Gerar malha (com faces)", value=True,
                help="Desative para gerar apenas nuvem de pontos (mais rapido).",
            )
        run = st.button("Gerar reconstrucao 3D", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        with st.spinner("Calculando coordenadas 3D e triangulando a superficie..."):
            st.session_state.mesh_result = reconstruct_3d(
                depth_result["depth_map"],
                depth_result["composite_image"],
                depth_result["confidence_map"],
                camera_height_mm,
                focus_distance_mm,
                focal_length_mm,
                sensor_width_mm,
                downsample,
                confidence_threshold,
                as_mesh,
            )

    mesh = st.session_state.mesh_result
    if mesh is None:
        st.info("Configure os parametros e gere a reconstrucao.")
        return

    gh, gw = mesh["grid_shape"]
    n_faces = len(mesh["faces"]) if mesh["faces"] else 0
    readout = (
        f"Nuvem gerada: <b>{len(mesh['vertices'])}</b> pontos "
        f"&nbsp;|&nbsp; grade <b>{gw}x{gh}</b> "
        f"&nbsp;|&nbsp; faces: <b>{n_faces}</b>"
    )
    st.markdown(f'<div class="console-panel console-readout">{readout}</div>', unsafe_allow_html=True)

    st.write("")
    if st.button("Avancar para visualizacao", type="primary"):
        unlock_next(4)
        st.session_state.current_step = 4
        st.rerun()