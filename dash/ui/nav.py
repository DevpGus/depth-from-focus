"""
ui/nav.py

Renderiza o trilho vertical de navegacao entre etapas (sidebar) e o cabecalho
de cada etapa no painel principal.
"""

from __future__ import annotations

import streamlit as st

from core.state import STEPS, go_to_step


def render_rail() -> None:
    st.markdown('<div class="console-eyebrow">Trilho de captura</div>', unsafe_allow_html=True)
    st.markdown('<div class="rail-track">', unsafe_allow_html=True)

    for i, step in enumerate(STEPS):
        state = "pending"
        if i == st.session_state.current_step:
            state = "active"
        elif i <= st.session_state.max_step_unlocked:
            state = "done"

        st.markdown(
            f'<div class="rail-node {state}">'
            f'<div class="rail-z">{step["z"]}</div>'
            f'<div class="rail-label">{step["label"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    for i, step in enumerate(STEPS):
        disabled = i > st.session_state.max_step_unlocked
        if st.button(
            f'{step["z"]} · {step["label"]}',
            key=f"nav_{step['key']}",
            disabled=disabled,
            use_container_width=True,
        ):
            go_to_step(i)
            st.rerun()


def render_step_header(title: str, z: str, description: str) -> None:
    st.markdown(f'<div class="console-eyebrow">{z} · Etapa</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.markdown(f'<p class="console-readout">{description}</p>', unsafe_allow_html=True)
    st.markdown("---")