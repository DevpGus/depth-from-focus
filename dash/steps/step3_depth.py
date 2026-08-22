from __future__ import annotations

import plotly.express as px
import streamlit as st

from core.pipeline import estimate_depth_map
from core.state import unlock_next
from ui.nav import render_step_header

PLOTLY_TEMPLATE = "plotly_dark"


def render() -> None:
    render_step_header(
        "Estimativa de profundidade",
        "Z02",
        "Para cada pixel, identifica em qual camada da pilha ele está mais nítido "
        "(shape-from-focus, magnitude do Laplaciano) e usa essa posição como profundidade.",
    )

    align_result = st.session_state.align_result
    if align_result is None:
        st.warning("Volte a etapa Z01 e rode o alinhamento.")
        return

    with st.container():
        st.markdown('<div class="console-panel">', unsafe_allow_html=True)
        blur = st.checkbox(
            "Suavizar mapa de nitidez (Gaussian 5x5) antes da seleção de foco",
            value=False,
            help="Reduz ruído pixel-a-pixel no mapa de profundidade, ao custo de alguma perda de detalhe fino.",
        )
        run = st.button("Calcular mapa de profundidade", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        with st.spinner("Calculando nitidez (Laplaciano) em cada fatia..."):
            st.session_state.depth_result = estimate_depth_map(align_result["aligned_images"], blur)

    result = st.session_state.depth_result
    if result is None:
        st.info("Configure os parametros e calcule o mapa de profundidade.")
        return

    tab_depth, tab_confidence, tab_composite = st.tabs(
        ["Mapa de profundidade", "Mapa de confianca", "Imagem all-in-focus"]
    )

    with tab_depth:
        fig = px.imshow(result["depth_map"], color_continuous_scale="Turbo", aspect="equal")
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title="Indice de fatia"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_confidence:
        fig = px.imshow(result["confidence_map"], color_continuous_scale="Viridis", aspect="equal")
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title="Nitidez (bruta)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_composite:
        st.image(
            result["composite_image"], channels="BGR", use_container_width=True,
            caption="Composicao all-in-focus",
        )

    st.write("")