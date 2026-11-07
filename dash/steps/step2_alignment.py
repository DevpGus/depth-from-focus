from __future__ import annotations

import streamlit as st

from core.pipeline import align_stack
from core.state import unlock_next
from ui.nav import render_step_header


def render() -> None:
    render_step_header(
        "Alinhamento",
        "Z01",
        "Alinhamento sequencial ECC (Enhanced Correlation Coefficient) com recorte "
        "automático para a região válida comum a toda a pilha.",
    )

    stack = st.session_state.stack_data
    if stack is None:
        st.warning("Volte a etapa Z00 e carregue uma pilha de imagens.")
        return

    st.markdown(
        '<div class="console-panel console-readout">'
        "Metodo: <b>ECC</b> (cv.findTransformECC) &nbsp;|&nbsp; "
        # "Modelo de transformacao: <b>Afim</b> &nbsp;|&nbsp; "
        "Cada fatia e alinhada contra a fatia imediatamente anterior, com a "
        "transformacao acumulada desde a primeira fatia."
        "</div>",
        unsafe_allow_html=True,
    )
    run = st.button("Rodar alinhamento", type="primary")

    if run:
        with st.spinner("Estimando transformacoes ECC entre fatias consecutivas..."):
            st.session_state.align_result = align_stack(stack["images"])

    result = st.session_state.align_result
    if result is None:
        st.info("Clique em 'Rodar alinhamento' para continuar.")
        return

    aligned = result["aligned_images"]
    st.markdown('<div class="console-eyebrow">Antes / depois</div>', unsafe_allow_html=True)
    preview_idx = st.select_slider("Fatia para comparar", options=list(range(len(aligned))), value=0)

    col_before, col_after = st.columns(2)
    with col_before:
        st.caption("Bruta")
        st.image(stack["images"][preview_idx], channels="BGR", use_container_width=True)
    with col_after:
        st.caption("Alinhada e recortada")
        st.image(aligned[preview_idx], channels="BGR", use_container_width=True)

    scale = result["scales"][preview_idx]
    acc_scale = result["acummulated_scales"][preview_idx]
    st.markdown(
        f'<div class="console-readout">Escala (zoom) estimada nesta fatia: <b>{scale:.4f}</b> '
        f"&nbsp;|&nbsp; Escala acumulada desde a fatia 0: <b>{acc_scale:.4f}</b></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    if st.button("Avançar para Estimativa de Profundidade", type="primary"):
        unlock_next(2)
        st.session_state.current_step = 2
        st.rerun()