from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from core.pipeline import export_point_cloud
from ui.nav import render_step_header


def render() -> None:
    render_step_header(
        "Visualizacao",
        "Z04",
        "Explore o modelo 3D reconstruido: rotacione, aproxime e exporte o resultado.",
    )

    mesh = st.session_state.mesh_result
    if mesh is None:
        st.warning("Volte a etapa Z03 e gere a reconstrucao 3D.")
        return

    vertices = mesh["vertices"]
    colors = mesh["colors"]
    faces = mesh["faces"]

    with st.container():
        st.markdown('<div class="console-panel">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            mode_options = ["Malha (superficie)", "Nuvem de pontos"] if faces else ["Nuvem de pontos"]
            render_mode = st.radio("Modo de renderizacao", mode_options, horizontal=True)
        with c2:
            color_mode = st.radio("Colorizacao", ["Textura real", "Profundidade (colormap)"], horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if render_mode == "Malha (superficie)" and faces:
        i = [f[0] for f in faces]
        j = [f[1] for f in faces]
        k = [f[2] for f in faces]
        if color_mode == "Textura real":
            vertexcolor = [f"rgb({r},{g},{b})" for r, g, b in colors]
            trace = go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=i, j=j, k=k, vertexcolor=vertexcolor, flatshading=False,
            )
        else:
            trace = go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=i, j=j, k=k, intensity=vertices[:, 2], colorscale="Turbo",
                colorbar=dict(title="Z (mm)"),
            )
    else:
        if color_mode == "Textura real":
            marker = dict(size=2, color=[f"rgb({r},{g},{b})" for r, g, b in colors])
        else:
            marker = dict(size=2, color=vertices[:, 2], colorscale="Turbo", colorbar=dict(title="Z (mm)"))
        trace = go.Scatter3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], mode="markers", marker=marker
        )

    fig = go.Figure(data=[trace])
    fig.update_layout(
        template="plotly_dark",
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode="data"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=620,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="console-eyebrow">Exportar modelo</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 3])
    with c1:
        fmt = st.selectbox("Formato", ["PLY", "XYZ"], label_visibility="collapsed")
    with c2:
        data = export_point_cloud(mesh, fmt)
        st.download_button(
            "Baixar modelo",
            data=data,
            file_name=f"reconstrucao_3d.{fmt.lower()}",
            mime="text/plain",
            type="primary",
        )