from __future__ import annotations

import streamlit as st

from core.pipeline import load_image_stack
from core.state import unlock_next
from core.synthetic import generate_synthetic_stack
from ui.nav import render_step_header


def render() -> None:
    render_step_header(
        "Captura da pilha",
        "Z00",
        "Envie as imagens capturadas ao longo do trilho vertical."
        " Formatos aceitos: PNG, JPG e TIFF.",
    )

    col_upload, col_demo = st.columns([3, 1])

    # with col_upload:
    files = st.file_uploader(
        "Imagens da pilha focal",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    # with col_demo:
    #     st.markdown('<div class="console-eyebrow">Sem imagens agora?</div>', unsafe_allow_html=True)
    #     use_demo = st.button("Usar pilha de exemplo", use_container_width=True)

    # if use_demo:
    #     demo_images = generate_synthetic_stack()
    #     st.session_state.stack_data = {
    #         "images": demo_images,
    #         "filenames": [f"exemplo_z{str(i).zfill(2)}.png" for i in range(len(demo_images))],
    #         "shape": demo_images[0].shape[:2],
    #         "n_slices": len(demo_images),
    #     }

    # elif files:
    if files:
        file_bytes = tuple(f.getvalue() for f in files)
        filenames = tuple(f.name for f in files)
        st.session_state.stack_data = load_image_stack(file_bytes, filenames)

    stack = st.session_state.stack_data

    if stack is None:
        st.info("Aguardando pilha de imagens para continuar.")
        return

    st.markdown(
        f'<div class="console-panel console-readout">'
        f"Pilha carregada: <b>{stack['n_slices']}</b> fatias &nbsp;|&nbsp; "
        f"Resolucao: <b>{stack['shape'][1]}x{stack['shape'][0]}</b> px"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="console-eyebrow">Pre-visualizacao</div>', unsafe_allow_html=True)
    thumb_cols = st.columns(6)
    for i, (img, name) in enumerate(zip(stack["images"], stack["filenames"])):
        with thumb_cols[i % 6]:
            st.image(img, channels="BGR", caption=name, use_container_width=True)
        if i >= 11 and stack["n_slices"] > 12:
            with thumb_cols[(i + 1) % 6]:
                st.markdown(
                    f'<div class="console-readout">+{stack["n_slices"] - 12} fatias</div>',
                    unsafe_allow_html=True,
                )
            break

    st.write("")
    if st.button("Avançar para Alinhamento", type="primary"):
        unlock_next(1)
        st.session_state.current_step = 1
        st.rerun()