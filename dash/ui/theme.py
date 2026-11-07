"""
ui/theme.py

Sistema de design do dashboard: "console de laboratorio".

Paleta (inspirada em monitores de MEV / osciloscopios de fosforo):
    bg-void        #0D1210   fundo principal (quase preto, subtom verde-frio)
    bg-panel       #161C19   paineis / cards
    bg-panel-raised#1E2622   elementos elevados (inputs, hover)
    accent-phosphor#7FE7C4   acao primaria / dado ativo
    accent-amber   #E3A857   atencao / posicao atual / destaque secundario
    text-primary   #E8EDE9
    text-muted     #8A9A92
    border-hairline#2A332E

Tipografia:
    Display -> Space Grotesk  (titulos, geometrica e tecnica)
    Corpo   -> IBM Plex Sans
    Dados   -> IBM Plex Mono  (parametros, coordenadas, nomes de arquivo)

Assinatura visual: a navegacao entre etapas e desenhada como um TRILHO
VERTICAL com marcacoes Z00-Z04, ecoando o proprio trilho fisico do
microscopio percorrido durante a captura da pilha focal.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg-void: #0D1210;
    --bg-panel: #161C19;
    --bg-panel-raised: #1E2622;
    --accent-phosphor: #7FE7C4;
    --accent-amber: #E3A857;
    --text-primary: #E8EDE9;
    --text-muted: #8A9A92;
    --border-hairline: #2A332E;
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-void);
}

section[data-testid="stSidebar"] {
    background: var(--bg-panel);
    border-right: 1px solid var(--border-hairline);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}

h1 { font-weight: 700 !important; }
h2, h3 { font-weight: 600 !important; }

.console-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent-phosphor);
    margin-bottom: 0.2rem;
}

.console-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border-hairline);
    border-radius: 3px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
}

.console-readout {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--text-muted);
}

.console-readout b, .console-readout strong {
    color: var(--accent-phosphor);
    font-weight: 500;
}

/* ---- Trilho vertical de navegacao (assinatura visual) -------------- */
.rail-track {
    position: relative;
    margin: 0.4rem 0 1.4rem 0.6rem;
    padding-left: 1.6rem;
    border-left: 2px solid var(--border-hairline);
}

.rail-node {
    position: relative;
    padding: 0.55rem 0 0.55rem 0.9rem;
    margin-left: -1.9px;
}

.rail-node::before {
    content: "";
    position: absolute;
    left: -1.72rem;
    top: 50%;
    transform: translateY(-50%);
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: var(--border-hairline);
    border: 2px solid var(--bg-panel);
}

.rail-node.done::before { background: var(--accent-phosphor); }
.rail-node.active::before {
    background: var(--accent-amber);
    box-shadow: 0 0 0 4px rgba(227, 168, 87, 0.18);
}

.rail-z {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.06em;
}

.rail-node.active .rail-z { color: var(--accent-amber); }
.rail-node.done .rail-z { color: var(--accent-phosphor); }

.rail-label {
    font-size: 0.9rem;
    color: var(--text-muted);
}

.rail-node.active .rail-label { color: var(--text-primary); font-weight: 600; }
.rail-node.done .rail-label { color: var(--text-primary); }

/* ---- Botoes ---------------------------------------------------------- */
.stButton > button {
    background: var(--bg-panel-raised);
    color: var(--text-primary);
    border: 1px solid var(--border-hairline);
    border-radius: 3px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
}

.stButton > button:hover {
    border-color: var(--accent-phosphor);
    color: var(--accent-phosphor);
}

.stButton > button[kind="primary"] {
    background: var(--accent-phosphor);
    color: #0D1210;
    border: none;
    font-weight: 600;
}

.stButton > button[kind="primary"]:hover {
    background: #93F0CC;
    color: #0D1210;
}

/* ---- Uploader, sliders, selects --------------------------------- */
[data-testid="stFileUploaderDropzone"] {
    background: var(--bg-panel-raised);
    border: 1px dashed var(--border-hairline);
}

.stSlider [data-baseweb="slider"] > div > div {
    background: var(--accent-phosphor) !important;
}

hr {
    border-color: var(--border-hairline);
}

#MainMenu, footer {visibility: hidden;}
</style>
"""


def inject(st_module) -> None:
    st_module.markdown(CSS, unsafe_allow_html=True)