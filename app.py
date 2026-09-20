from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import tempfile
import base64
import seaborn as sns
import matplotlib.pyplot as plt
from g4f.client import Client
from langchain.embeddings.base import Embeddings
from typing import List
from ydata_profiling import ProfileReport
from bs4 import BeautifulSoup

# Custom embedding class to match the expected interface
class CustomEmbedding(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0]


# Load Sentence-Transformers model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
custom_embedding = CustomEmbedding(embedding_model)


def read_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ")
    return text


# ============================================================
# ADDED CODE: UI styling (HTML / CSS only, no app logic)
# ============================================================
# Design notes
#   - Idea: a highlighter pen on graph paper. One accent colour
#     (marigold) is used the way you would highlight a spreadsheet:
#     active tab, main button, focus, your own chat messages.
#   - Type: Archivo (Google Fonts). Wide and heavy for the title,
#     normal width everywhere else.
#   - Neutrals are semi-transparent slate overlays, so the same CSS
#     works with Streamlit's light and dark themes.
#   - Motion: only the loading bars (shown while a report is being
#     generated) and a short fade on new chat messages.

APP_CSS = r"""<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap');

:root {
    --hl: #FFB400;
    --hl-hover: #F0A500;
    --hl-wash: rgba(255, 180, 0, 0.13);
    --hl-line: rgba(255, 180, 0, 0.55);
    --ink: #161B2C;
    --slate: #3A4460;
    --bar: #7A879F;
    --rule: rgba(112, 124, 150, 0.34);
    --rule-soft: rgba(112, 124, 150, 0.2);
    --wash: rgba(112, 124, 150, 0.09);
    --grid: rgba(112, 124, 150, 0.14);
    --r-frame: 16px;
    --r-control: 12px;
    --font: 'Archivo', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;

    /* line icons, used as masks so they take the colour of the text */
    --i-upload: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 16V4M6.5 9.5 12 4l5.5 5.5M4 20h16'/%3E%3C/svg%3E");
    --i-download: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 4v12M6.5 10.5 12 16l5.5-5.5M4 20h16'/%3E%3C/svg%3E");
    --i-chart: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M5 20v-8M12 20V4M19 20v-5M3 20h18'/%3E%3C/svg%3E");
    --i-table: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3.5' y='4.5' width='17' height='15' rx='2'/%3E%3Cpath d='M3.5 10h17M3.5 15h17M9.5 4.5v15'/%3E%3C/svg%3E");
    --i-scatter: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 4v16h16'/%3E%3Ccircle cx='9' cy='14' r='1.3' fill='black'/%3E%3Ccircle cx='13' cy='9' r='1.3' fill='black'/%3E%3Ccircle cx='17.5' cy='13' r='1.3' fill='black'/%3E%3C/svg%3E");
    --i-chat: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M20.5 11.5a8 8 0 0 1-11.6 7.1L4 20l1.4-4.6A8 8 0 1 1 20.5 11.5Z'/%3E%3C/svg%3E");
}

/* ---------- Base typography (icons and code keep their own fonts) ---------- */
.stApp,
.stApp :where(*):not(code):not(pre):not(pre *):not(code *):not([data-testid="stIconMaterial"]) {
    font-family: var(--font);
}
.stApp {
    line-height: 1.55;
    font-variant-numeric: tabular-nums;
    -webkit-font-smoothing: antialiased;
}
.stApp * {
    scrollbar-width: thin;
    scrollbar-color: rgba(112, 124, 150, 0.5) transparent;
}
::selection { background: var(--hl); color: var(--ink); }
.stApp :focus-visible { outline: 2px solid var(--hl); outline-offset: 2px; }

/* ---------- Page frame ---------- */
.stApp [data-testid="stMainBlockContainer"],
.stApp .block-container {
    position: relative;
    z-index: 1;
    max-width: 1180px !important;
    padding: 3.5rem 2.75rem 8rem !important;
}
.stApp [data-testid="stAppDeployButton"],
.stApp [data-testid="stDeployButton"],
.stApp [data-testid="stDecoration"],
.stApp footer { display: none !important; }

/* the element that only carries this <style> block should not take up space */
.stApp .element-container:has(style) { display: none !important; }

/* graph-paper grid behind the top of the page, fading out */
.stApp section.main,
.stApp section.stMain,
.stApp [data-testid="stMain"] { position: relative; }
.stApp section.main::before,
.stApp section.stMain::before,
.stApp [data-testid="stMain"]::before {
    content: "";
    position: absolute;
    z-index: 0;
    top: 0;
    left: 0;
    right: 0;
    height: 460px;
    pointer-events: none;
    background-image:
        linear-gradient(to right, var(--grid) 1px, transparent 1px),
        linear-gradient(to bottom, var(--grid) 1px, transparent 1px);
    background-size: 44px 44px;
    -webkit-mask-image: radial-gradient(ellipse 75% 100% at 10% 0%, #000, transparent 72%);
    mask-image: radial-gradient(ellipse 75% 100% at 10% 0%, #000, transparent 72%);
}

/* ---------- Title: histogram mark + wide, heavy type ---------- */
.stApp h1 {
    display: flex;
    align-items: center;
    gap: 0.42em;
    padding: 0;
    margin: 0;
    font-size: clamp(1.75rem, 0.9rem + 3vw, 3.2rem);
    font-weight: 800;
    font-stretch: 125%;
    line-height: 1.02;
    letter-spacing: -0.025em;
}
.stApp h1::before {
    content: "";
    flex: none;
    width: 0.98em;
    height: 0.7em;
    background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 56 40'%3E%3Cg fill='%237A879F'%3E%3Crect x='0' y='30' width='6' height='10' rx='1.5'/%3E%3Crect x='8' y='22' width='6' height='18' rx='1.5'/%3E%3Crect x='16' y='12' width='6' height='28' rx='1.5'/%3E%3Crect x='32' y='10' width='6' height='30' rx='1.5'/%3E%3Crect x='40' y='20' width='6' height='20' rx='1.5'/%3E%3Crect x='48' y='29' width='6' height='11' rx='1.5'/%3E%3C/g%3E%3Crect x='24' y='0' width='6' height='40' rx='1.5' fill='%23FFB400'/%3E%3C/svg%3E") center / contain no-repeat;
}
.stApp [data-testid="stHeaderActionElements"] { display: none !important; }
.stApp .hero-tagline {
    max-width: 62ch;
    margin: 0.65rem 0 1.75rem;
    font-size: 1.08rem;
    opacity: 0.72;
}

/* Section titles inside the tabs are smaller and have no mark */
.stApp [role="tabpanel"] h1 {
    display: block;
    padding: 0.6rem 0 0.35rem;
    font-size: 1.55rem;
    font-weight: 700;
    font-stretch: 100%;
    line-height: 1.2;
    letter-spacing: -0.012em;
}
.stApp [role="tabpanel"] h1::before { content: none; }

/* ---------- Tabs ---------- */
.stApp [role="tablist"] { gap: 0.35rem; }
.stApp [role="tablist"]::after { height: 1px !important; background: var(--rule) !important; }
.stApp [role="tab"] {
    padding: 0.7rem 1.15rem;
    border-radius: 10px 10px 0 0;
    color: inherit;
    transition: background-color 0.15s ease;
}
.stApp [role="tab"] p {
    margin: 0;
    font-size: 0.98rem;
    font-weight: 500;
    color: inherit !important;
    opacity: 0.66;
}
.stApp [role="tab"] p::before {
    content: "";
    display: inline-block;
    width: 1.1em;
    height: 1.1em;
    margin-right: 0.5em;
    vertical-align: -0.2em;
    background-color: currentColor;
    -webkit-mask: var(--i) center / contain no-repeat;
    mask: var(--i) center / contain no-repeat;
}
.stApp [role="tab"]:nth-child(1) p { --i: var(--i-chart); }
.stApp [role="tab"]:nth-child(2) p { --i: var(--i-table); }
.stApp [role="tab"]:nth-child(3) p { --i: var(--i-scatter); }
.stApp [role="tab"]:hover { background: var(--wash); }
.stApp [role="tab"]:hover p { opacity: 1; }
.stApp [role="tab"][aria-selected="true"],
.stApp [role="tab"][aria-selected="true"] p {
    color: inherit !important;
    opacity: 1;
    font-weight: 700;
}
.stApp .react-aria-SelectionIndicator,
.stApp [data-baseweb="tab-highlight"] {
    background: var(--hl) !important;
    height: 3px !important;
    border-radius: 3px 3px 0 0;
}
.stApp [data-baseweb="tab-border"] { background: var(--rule) !important; }

/* ---------- Sidebar ---------- */
.stApp [data-testid="stSidebar"] { border-right: 1px solid var(--rule-soft); }
.stApp [data-testid="stSidebar"] h2 {
    padding: 0.25rem 0 0.5rem;
    font-size: 1.2rem;
    font-weight: 700;
    font-stretch: 100%;
    letter-spacing: -0.01em;
}
.stApp [data-testid="stSidebar"] h2::before {
    content: "";
    display: inline-block;
    width: 1.05em;
    height: 1.05em;
    margin-right: 0.45em;
    vertical-align: -0.14em;
    background-color: currentColor;
    -webkit-mask: var(--i-upload) center / contain no-repeat;
    mask: var(--i-upload) center / contain no-repeat;
}
.stApp [data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed var(--rule);
    border-radius: var(--r-frame);
    background: var(--wash);
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.stApp [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--hl);
    background: var(--hl-wash);
}
.stApp [data-testid="stFileUploaderDropzone"] button {
    border: 1px solid var(--rule);
    border-radius: 10px;
}

/* Buttons (Download Report) */
.stApp .stButton > button {
    padding: 0.55rem 1.15rem;
    border: 0;
    border-radius: 10px;
    background: var(--hl);
    color: var(--ink) !important;
    font-weight: 700;
    transition: background-color 0.15s ease;
}
.stApp .stButton > button p {
    color: var(--ink) !important;
    font-weight: 700;
}
.stApp .stButton > button p::before {
    content: "";
    display: inline-block;
    width: 1.05em;
    height: 1.05em;
    margin-right: 0.5em;
    vertical-align: -0.16em;
    background-color: currentColor;
    -webkit-mask: var(--i-download) center / contain no-repeat;
    mask: var(--i-download) center / contain no-repeat;
}
.stApp .stButton > button:hover { background: var(--hl-hover); }
.stApp .stButton > button:active { transform: translateY(1px); }

/* Download link that appears after clicking the button */
.stApp a.dl-link {
    display: block;
    padding: 0.6rem 1rem;
    border: 1.5px solid var(--hl);
    border-radius: 10px;
    background: var(--hl-wash);
    color: inherit !important;
    font-weight: 600;
    text-align: center;
    text-decoration: none !important;
    transition: background-color 0.15s ease;
}
.stApp a.dl-link:hover { background: var(--hl); color: var(--ink) !important; }

/* Progress bar */
.stApp [data-testid="stProgressBarTrack"] { background: var(--wash); border-radius: 99px; }
.stApp [data-testid="stProgressBarTrack"] > div { background: var(--hl) !important; border-radius: 99px; }
.stApp .stProgress > div > div > div > div { background-color: var(--hl) !important; }

/* ---------- Loading state: bars that rise and fall like a live histogram ---------- */
@property --l1 { syntax: "<length>"; inherits: false; initial-value: 8px; }
@property --l2 { syntax: "<length>"; inherits: false; initial-value: 16px; }
@property --l3 { syntax: "<length>"; inherits: false; initial-value: 26px; }
@property --l4 { syntax: "<length>"; inherits: false; initial-value: 16px; }
@property --l5 { syntax: "<length>"; inherits: false; initial-value: 8px; }
@keyframes bar-1 { 0%, 100% { --l1: 6px; }  50% { --l1: 24px; } }
@keyframes bar-2 { 0%, 100% { --l2: 8px; }  50% { --l2: 26px; } }
@keyframes bar-3 { 0%, 100% { --l3: 12px; } 50% { --l3: 28px; } }
@keyframes bar-4 { 0%, 100% { --l4: 8px; }  50% { --l4: 26px; } }
@keyframes bar-5 { 0%, 100% { --l5: 6px; }  50% { --l5: 24px; } }

.stApp [data-testid="stSpinner"] > div {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1.15rem;
    border: 1px solid var(--rule-soft);
    border-radius: var(--r-control);
    background: var(--wash);
}
.stApp [data-testid="stSpinner"] > div > span,
.stApp [data-testid="stSpinner"] i { display: none !important; }
.stApp [data-testid="stSpinner"] p { margin: 0; font-weight: 600; }
.stApp [data-testid="stSpinner"] > div::before {
    content: "";
    flex: none;
    width: 36px;
    height: 28px;
    background:
        linear-gradient(var(--bar) 0 0) 0 100% / 4px var(--l1) no-repeat,
        linear-gradient(var(--bar) 0 0) 8px 100% / 4px var(--l2) no-repeat,
        linear-gradient(var(--hl) 0 0) 16px 100% / 4px var(--l3) no-repeat,
        linear-gradient(var(--bar) 0 0) 24px 100% / 4px var(--l4) no-repeat,
        linear-gradient(var(--bar) 0 0) 32px 100% / 4px var(--l5) no-repeat;
}
@media (prefers-reduced-motion: no-preference) {
    .stApp [data-testid="stSpinner"] > div::before {
        animation:
            bar-1 1.3s ease-in-out infinite,
            bar-2 1.3s -0.26s ease-in-out infinite,
            bar-3 1.3s -0.52s ease-in-out infinite,
            bar-4 1.3s -0.78s ease-in-out infinite,
            bar-5 1.3s -1.04s ease-in-out infinite;
    }
}

/* ---------- Content blocks ---------- */
/* The profile report sits in a framed panel. */
.stApp .element-container:has(> iframe) {
    overflow: hidden;
    border: 1px solid var(--rule);
    border-radius: var(--r-frame);
    background: #ffffff;
}
.stApp .element-container > iframe { display: block; border: 0 !important; }
.stApp [data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid var(--rule);
    border-radius: var(--r-frame);
}
.stApp [data-testid="stImage"] {
    border: 1px solid var(--rule);
    border-radius: var(--r-frame);
    background: #ffffff;
    line-height: 0;
}
.stApp [data-testid="stImage"] img { border-radius: calc(var(--r-frame) - 1px); }

/* Dark theme: the white report and plots become dark panels instead of glaring slabs. */
@media (prefers-color-scheme: dark) {
    .stApp iframe[data-testid="stIFrame"],
    .stApp iframe.stIFrame,
    .stApp [role="tabpanel"] [data-testid="stImage"] img { filter: invert(0.92) hue-rotate(180deg); }
    .stApp .element-container:has(> iframe),
    .stApp [role="tabpanel"] [data-testid="stImage"] { background: #141414; }
}

/* inline code (e.g. column names in chat answers): neutral chip instead of green text */
.stApp [data-testid="stMarkdownContainer"] :not(pre) > code {
    padding: 0.12em 0.4em;
    border: 1px solid var(--rule-soft);
    border-radius: 6px;
    background: var(--wash);
    color: inherit;
    font-size: 0.88em;
}

/* ---------- Alerts ---------- */
.stApp [data-testid="stAlertContainer"],
.stApp [data-testid="stNotification"],
.stApp [data-testid="stAlert"] > div {
    border: 1px solid var(--rule-soft);
    border-radius: var(--r-control);
}
/* info boxes get the highlighter treatment (newer and older Streamlit use different test ids) */
.stApp [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]),
.stApp [data-testid="stNotification"]:has([data-testid="stNotificationContentInfo"]) {
    background: var(--hl-wash);
    border-color: var(--hl-line);
    box-shadow: inset 4px 0 0 var(--hl);
    color: inherit;
}
.stApp [data-testid="stAlertContentInfo"],
.stApp [data-testid="stAlertContentInfo"] p,
.stApp [data-testid="stNotificationContentInfo"],
.stApp [data-testid="stNotificationContentInfo"] p { color: inherit; }

/* ---------- Empty page (before a file is uploaded) ---------- */
/* nothing to switch between yet, so the tab bar is hidden */
.stApp:has(.steps) [data-testid="stTabs"],
.stApp:has(.steps) .stTabs { display: none !important; }
.stApp [data-testid="stHorizontalBlock"]:has(.steps) { align-items: center; }
.stApp .steps {
    margin: 1.75rem 0 0;
    padding: 0;
    list-style: none;
    counter-reset: step;
}
.stApp .steps li {
    position: relative;
    margin: 0;
    padding: 0 0 1.5rem 3.2rem;
    counter-increment: step;
}
.stApp .steps li::before {
    content: counter(step);
    position: absolute;
    left: 0;
    top: 0;
    display: grid;
    place-items: center;
    width: 2.15rem;
    height: 2.15rem;
    border-radius: 50%;
    background: var(--hl);
    color: var(--ink);
    font-weight: 700;
}
.stApp .steps li:not(:last-child)::after {
    content: "";
    position: absolute;
    left: 1.03rem;
    top: 2.45rem;
    bottom: 0.3rem;
    width: 1px;
    background: var(--rule);
}
.stApp .steps strong {
    display: block;
    font-size: 1.08rem;
    font-weight: 700;
}
.stApp .steps span { opacity: 0.72; }

/* ---------- Chat ---------- */
.stApp .chat-head {
    display: flex;
    align-items: center;
    gap: 0.6em;
    margin: 2.5rem 0 0.35rem;
    padding-top: 1.4rem;
    border-top: 1px solid var(--rule);
    font-size: 1.25rem;
    font-weight: 700;
    font-stretch: 100%;
    letter-spacing: -0.01em;
}
.stApp .chat-head::before {
    content: "";
    width: 1.15em;
    height: 1.15em;
    background-color: currentColor;
    -webkit-mask: var(--i-chat) center / contain no-repeat;
    mask: var(--i-chat) center / contain no-repeat;
}
.stApp .chat-hint { margin: 0 0 1.1rem; opacity: 0.68; }
.stApp [data-testid="stChatMessage"] {
    gap: 0.9rem;
    margin-bottom: 0.75rem;
    padding: 1rem 1.15rem;
    border: 1px solid var(--rule-soft);
    border-radius: var(--r-frame);
    background: var(--wash);
}
/* (older Streamlit versions call the avatars chatAvatarIcon-user / chatAvatarIcon-assistant) */
.stApp [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
.stApp [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-color: var(--hl-line);
    background: var(--hl-wash);
}
.stApp [data-testid="stChatMessageAvatarUser"],
.stApp [data-testid="chatAvatarIcon-user"] {
    background: var(--hl) !important;
    color: var(--ink) !important;
    border-radius: 10px;
}
.stApp [data-testid="stChatMessageAvatarAssistant"],
.stApp [data-testid="chatAvatarIcon-assistant"] {
    background: var(--slate) !important;
    color: #ffffff !important;
    border-radius: 10px;
}
.stApp [data-testid="stBottomBlockContainer"] {
    max-width: 1180px !important;
    padding: 1rem 2.75rem 1.5rem !important;
}
.stApp [data-testid="stChatInput"] {
    border-radius: 14px;
    background: transparent !important;
}
.stApp [data-testid="stChatInput"] > div {
    border: 1px solid var(--rule);
    border-radius: 14px;
    background: var(--wash);
}
.stApp [data-testid="stChatInput"] > div:focus-within {
    border-color: var(--hl);
    box-shadow: 0 0 0 3px rgba(255, 180, 0, 0.28);
}
/* older Streamlit versions draw a second (red) focus border one level deeper */
.stApp [data-testid="stChatInput"] > div > div {
    border-color: transparent !important;
    box-shadow: none !important;
    background: transparent !important;
}
.stApp [data-testid="stChatInputTextArea"]:focus-visible { outline: none; }
.stApp [data-testid="stChatInputSubmitButton"]:not(:disabled) {
    background: var(--hl);
    color: var(--ink);
}

/* ---------- Motion: only new chat messages fade in, and only if allowed ---------- */
@media (prefers-reduced-motion: no-preference) {
    .stApp [data-testid="stChatMessage"] { animation: message-in 0.22s ease-out; }
    @keyframes message-in { from { opacity: 0; } to { opacity: 1; } }
}

/* ---------- Small screens ---------- */
@media (max-width: 640px) {
    .stApp [data-testid="stMainBlockContainer"],
    .stApp .block-container { padding: 4.5rem 1rem 7rem !important; }
    .stApp [data-testid="stBottomBlockContainer"] { padding: 0.75rem 1rem 1rem !important; }
    .stApp [role="tab"] { padding: 0.6rem 0.55rem; }
    .stApp [role="tab"] p { font-size: 0.9rem; }
    .stApp [role="tab"] p::before { margin-right: 0.35em; }
    .stApp [data-testid="stHorizontalBlock"]:has(.steps) { row-gap: 1.5rem; }
}
@media (max-width: 420px) {
    .stApp [role="tab"] p::before { display: none; }
}
</style>"""

HERO_TAGLINE = (
    '<p class="hero-tagline">Profile a CSV or Excel file, explore it, '
    'and ask it questions.</p>'
)

EMPTY_STATE_STEPS = (
    '<ol class="steps">'
    '<li><strong>Upload a file</strong><span>CSV, XLS or XLSX, from the sidebar.</span></li>'
    '<li><strong>Review your data</strong><span>Profile report, table view and pair plots.</span></li>'
    '<li><strong>Ask questions</strong><span>Chat with your data in plain language.</span></li>'
    '</ol>'
)

CHAT_HEADING = (
    '<div class="chat-head">Chat with your data</div>'
    '<p class="chat-hint">Ask about columns, missing values, distributions or duplicates.</p>'
)


def apply_custom_style():
    st.markdown(APP_CSS, unsafe_allow_html=True)



# ============================================================
# ADDED CODE: Direct Pandas Dataset Analysis
# ============================================================

def analyze_dataset(prompt, df):
    prompt_lower = prompt.lower()

    # --------------------------------------------------------
    # Total number of customers
    # --------------------------------------------------------
    if (
        ("how many customers" in prompt_lower)
        and ("churn" in prompt_lower)
    ):
        total = len(df)
        churned = (df["Churn"] == "Yes").sum()
        churn_rate = (churned / total) * 100

        return (
            f"Total customers: {total}\n"
            f"Churned customers: {churned}\n"
            f"Churn rate: {churn_rate:.2f}%"
        )

    # --------------------------------------------------------
    # Average MonthlyCharges of churned customers
    # --------------------------------------------------------
    if (
        (
            "average monthly charge" in prompt_lower
            or "average monthlycharges" in prompt_lower
            or "mean monthly charge" in prompt_lower
        )
        and "churn" in prompt_lower
    ):
        churned_customers = df[df["Churn"] == "Yes"]

        average_charge = churned_customers["MonthlyCharges"].mean()

        return (
            f"Number of churned customers: {len(churned_customers)}\n"
            f"Average MonthlyCharges of churned customers: "
            f"${average_charge:.2f}"
        )

    # --------------------------------------------------------
    # Percentage of month-to-month customers
    # --------------------------------------------------------
    if (
        "month-to-month" in prompt_lower
        and "contract" in prompt_lower
        and "percentage" in prompt_lower
    ):
        count = (df["Contract"] == "Month-to-month").sum()
        total = len(df)
        percentage = (count / total) * 100

        return (
            f"Month-to-month customers: {count}\n"
            f"Total customers: {total}\n"
            f"Percentage: {percentage:.2f}%"
        )

    # --------------------------------------------------------
    # Highest churn rate by contract
    # --------------------------------------------------------
    if (
        "highest churn" in prompt_lower
        and "contract" in prompt_lower
    ):
        churn_rates = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
        )

        highest_contract = churn_rates.idxmax()
        highest_rate = churn_rates.max()

        result = "Churn rate by contract:\n"

        for contract, rate in churn_rates.items():
            result += f"{contract}: {rate:.2f}%\n"

        result += (
            f"\nHighest churn rate: {highest_contract} "
            f"({highest_rate:.2f}%)"
        )

        return result

    # --------------------------------------------------------
    # Overall churn rate
    # --------------------------------------------------------
    if (
        "churn rate" in prompt_lower
        and "contract" not in prompt_lower
    ):
        churn_rate = (df["Churn"] == "Yes").mean() * 100

        return f"Overall churn rate: {churn_rate:.2f}%"

    # No direct analysis available
    return None


def main():
    st.set_page_config(
        page_title="Data Analyzer Ai",
        page_icon="📊",
        layout="wide"
    )
    apply_custom_style()

    st.title("AI-Based Data Analyzer")
    st.markdown(HERO_TAGLINE, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(
        ["Data Profile", "Excel Sheet", "Pair Plots"]
    )

    # File uploader
    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file",
        type=["xls", "xlsx", "csv"]
    )

    if uploaded_file is None:
        left_col, right_col = st.columns([5, 6], gap="large")

        with left_col:
            st.info(
                "Please upload a file of type: "
                + ", ".join(["xls", "xlsx", "csv"])
                + " to start analysing your data."
            )
            st.markdown(EMPTY_STATE_STEPS, unsafe_allow_html=True)

        with right_col:
            st.image("waiting.jpg", use_column_width=True)

        return

    elif uploaded_file is not None:
        st.sidebar.caption(
            "Uploaded file: " + uploaded_file.name
        )

        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)

        else:
            st.error(
                "Unsupported file format. "
                "Please upload a CSV or Excel file."
            )
            return

        # Generate ProfileReport if not already generated
        if "profile_report_path" not in st.session_state:
            with st.spinner('Generating Profile Report...'):
                profile = ProfileReport(
                    df,
                    correlations={
                        "auto": {
                            "calculate": False
                        }
                    },
                    missing_diagrams={
                        "Heatmap": False
                    }
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix='.html'
                ) as temp_file:
                    profile.to_file(
                        output_file=temp_file.name
                    )
                    st.session_state.profile_report_path = (
                        temp_file.name
                    )

            # Extract text from ProfileReport HTML
            with open(
                st.session_state.profile_report_path,
                'r'
            ) as f:
                html_string = f.read()
                profile_text = read_html(html_string)

            # # Generate embeddings for the profile report text
            # profile_text_list = [profile_text]
            # profile_embeddings = custom_embedding.embed_documents(profile_text_list)

            # # Store embeddings in FAISS
            # vectorstore = FAISS.from_texts(
            #     texts=[profile_text],
            #     embedding=custom_embedding
            # )
            # st.session_state.vectorstore = vectorstore

            # Split profile report into smaller chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000,
                chunk_overlap=300
            )

            profile_chunks = text_splitter.split_text(
                profile_text
            )

            # Store chunks in FAISS
            vectorstore = FAISS.from_texts(
                texts=profile_chunks,
                embedding=custom_embedding
            )

            st.session_state.vectorstore = vectorstore

        else:
            # Load the existing profile report HTML content
            with open(
                st.session_state.profile_report_path,
                'r'
            ) as f:
                html_string = f.read()

        # Initialize G4F chatbot client if not already initialized
        if "client" not in st.session_state:
            try:
                st.session_state.client = Client()

            except Exception as e:
                st.error(
                    f"Error initializing the AI chatbot: {e}"
                )
                st.session_state.client = None

        with tab1:
            st.title("Data Profile")

            if 'html_string' in locals():
                styled_html = (
                    f'<div style="background: #ffffff; color: #1f2430; '
                    f'padding: 12px 20px 24px;">{html_string}</div>'
                )

                st.components.v1.html(
                    styled_html,
                    height=800,
                    scrolling=True
                )

            else:
                st.warning(
                    "Data Profile is not available."
                )

        with tab2:
            st.title("Excel")
            st.dataframe(df)

        st.markdown(CHAT_HEADING, unsafe_allow_html=True)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input(
            "Ask AI anything regarding your data?"
        ):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

            with st.chat_message("user"):
                st.markdown(prompt)

            if st.session_state.client:
                try:

                    # ====================================================
                    # ADDED CODE: Direct Pandas Analysis
                    # ====================================================

                    analysis_result = analyze_dataset(
                        prompt,
                        df
                    )

                    if analysis_result is not None:

                        analysis_prompt = f"""
You are a data analysis assistant.

The following result was calculated directly from the
uploaded dataset using Pandas:

{analysis_result}

User question:
{prompt}

Explain the result clearly and concisely.

IMPORTANT:
- Do not change any numerical value.
- Do not invent additional numerical values.
- Use only the calculated result provided above.
"""

                        analysis_response = (
                            st.session_state.client
                            .chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {
                                        "role": "user",
                                        "content": analysis_prompt
                                    }
                                ]
                            )
                        )

                        analysis_answer = (
                            analysis_response
                            .choices[0]
                            .message
                            .content
                        )

                        with st.chat_message("assistant"):
                            analysis_placeholder = st.empty()

                            analysis_placeholder.markdown(
                                analysis_answer + "|"
                            )

                        analysis_placeholder.markdown(
                            analysis_answer
                        )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": analysis_answer
                            }
                        )

                        # Stop here so the existing FAISS code
                        # does not run for this question.
                        st.stop()

                    # ====================================================
                    # END OF ADDED CODE
                    # ====================================================

                    # Query FAISS vectorstore for the relevant data
                    # based on user prompt

                    # retriever = st.session_state.vectorstore.as_retriever()
                    # docs = retriever.get_relevant_documents(prompt)

                    retriever = (
                        st.session_state.vectorstore
                        .as_retriever(
                            search_kwargs={"k": 3}
                        )
                    )

                    docs = retriever.get_relevant_documents(
                        prompt
                    )

                    # Generate a context from the retrieved documents
                    context = "\n".join(
                        [
                            doc.page_content
                            for doc in docs
                        ]
                    )

                    full_prompt = (
                        f"Context: {context}\n\n"
                        f"User Prompt: {prompt}\n\n"
                        "Please respond in English with specific "
                        "insights or answers based on the dataset "
                        "provided."
                    )

                    response = (
                        st.session_state.client
                        .chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                    "role": "user",
                                    "content": full_prompt
                                }
                            ]
                        )
                    )

                    full_response = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()

                        message_placeholder.markdown(
                            full_response + "|"
                        )

                    message_placeholder.markdown(
                        full_response
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response
                        }
                    )

                except Exception as e:
                    st.error(
                        f"Error during chat interaction: {e}"
                    )

            else:
                st.error(
                    "G4F chatbot is not available due to "
                    "initialization error."
                )

        html_button_sidebar = st.sidebar.button(
            "Download Report"
        )

        if html_button_sidebar:
            progress_bar = st.sidebar.progress(0)
            progress_message = st.sidebar.empty()

            progress_message.text(
                "Preparing profile report for download..."
            )
            progress_bar.progress(10)

            # Use the previously generated ProfileReport
            # for download
            with open(
                st.session_state.profile_report_path,
                "rb"
            ) as f:
                report_data = f.read()

            progress_bar.progress(30)

            encoded_report = base64.b64encode(
                report_data
            ).decode()

            download_link = (
                f'<a class="dl-link" href="data:text/html;base64,'
                f'{encoded_report}" '
                f'download="profile_report.html">'
                f'Click here to download the profile report'
                f'</a>'
            )

            progress_bar.progress(60)

            st.sidebar.markdown(
                download_link,
                unsafe_allow_html=True
            )

            progress_bar.progress(80)

            st.sidebar.info(
                "☝︎ Click the link to download the profile "
                "report, and open it in your browser.\n\n"
                "You can press Ctrl+P to save the report as a PDF."
            )

            progress_bar.progress(100)

            progress_message.text(
                "Download link generated successfully!"
            )

        numeric_columns = df.select_dtypes(
            include=['number']
        )

        non_blank_numeric_columns = (
            numeric_columns.dropna(axis=1)
        )

        if not non_blank_numeric_columns.empty:
            with tab3:
                st.title("Pair Plots")

                with st.spinner(
                    'Generating Pair Plot...'
                ):
                    try:
                        fig = sns.pairplot(
                            non_blank_numeric_columns
                        )

                        st.pyplot(fig)

                        st.markdown(
                            "Pair Plot Generated on these columns: \n\n"
                        )

                        st.write(
                            non_blank_numeric_columns
                        )

                    except Exception as e:
                        st.error(
                            "An error occurred while generating "
                            f"the pair plot: {e}"
                        )
                        st.stop()

        else:
            with tab3:
                st.warning(
                    "No numeric value column found for "
                    "pair plot generation."
                )


if __name__ == "__main__":
    main()