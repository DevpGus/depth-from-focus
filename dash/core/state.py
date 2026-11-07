"""
core/state.py

Centraliza a inicializacao e a navegacao do st.session_state. Manter esses
helpers em um so lugar evita chaves de sessao espalhadas e duplicadas pelos
arquivos de steps/.
"""

from __future__ import annotations

import streamlit as st

STEPS = [
    {"key": "upload", "label": "Captura", "z": "Z00"},
    {"key": "alignment", "label": "Alinhamento", "z": "Z01"},
    {"key": "depth", "label": "Profundidade", "z": "Z02"},
    # {"key": "reconstruction", "label": "Reconstrucao 3D", "z": "Z03"},
    # {"key": "visualization", "label": "Visualizacao", "z": "Z04"},
]


def init_state() -> None:
    defaults = {
        "current_step": 0,
        "max_step_unlocked": 0,
        "stack_data": None,
        "align_result": None,
        "depth_result": None,
        "mesh_result": None,
        "align_params": {"method": "Correlacao de fase", "reference_index": 0},
        "depth_params": {"method": "Variancia do Laplaciano", "window_size": 9},
        "recon_params": {"downsample": 3, "z_scale": 40.0, "smoothing": 1.5},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_step(index: int) -> None:
    index = max(0, min(index, len(STEPS) - 1))
    st.session_state.current_step = index


def unlock_next(index: int) -> None:
    st.session_state.max_step_unlocked = max(st.session_state.max_step_unlocked, index)


def reset_pipeline() -> None:
    for key in ["stack_data", "align_result", "depth_result", "mesh_result"]:
        st.session_state[key] = None
    st.session_state.current_step = 0
    st.session_state.max_step_unlocked = 0