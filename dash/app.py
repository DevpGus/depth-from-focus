from __future__ import annotations

import streamlit as st

from core.state import STEPS, init_state, reset_pipeline
from steps import (
    step1_upload,
    step2_alignment,
    step3_depth,
    step4_reconstruction,
    step5_visualization,
)
from ui import theme
from ui.nav import render_rail

st.set_page_config(
    page_title="Reconstrucao 3D · Pilha Focal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject(st)
init_state()

STEP_RENDERERS = [
    step1_upload.render,
    step2_alignment.render,
    step3_depth.render,
]

with st.sidebar:
    st.markdown("### ◈ Análise Depth from Focus")
    st.markdown(
        '<p class="console-readout">Imagens &rarr; Alinhamento &rarr; Profundidade</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    render_rail()
    st.markdown("---")
    if st.button("Reiniciar pipeline", use_container_width=True):
        reset_pipeline()
        st.rerun()

STEP_RENDERERS[st.session_state.current_step]()