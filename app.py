"""Streamlit frontend for the AI-Powered Renewable Energy Knowledge Assistant."""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.document_processor import (
    RENEWABLE_CATEGORIES,
    DocumentProcessor,
    PDFProcessingError,
    TextEncodingError,
)
from modules.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingManager
from modules.rag_pipeline import DEFAULT_GEMINI_MODEL, RAGPipeline
from modules.vector_database import VectorDatabase

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "documents"
CHROMA_DIR = BASE_DIR / "chroma_db"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, encoding="utf-8", override=False)

NAVIGATION_PAGES = ["Home", "Knowledge Base", "Ask Assistant", "About"]

MODEL_OPTIONS = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
]

CORE_OVERVIEW_DOMAINS: list[dict[str, str]] = [
    {
        "name": "Solar Energy",
        "desc": "Photovoltaic cell architectures, MPPT tracking, bifacial arrays, and concentrated solar thermal systems.",
        "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>',
    },
    {
        "name": "Wind Energy",
        "desc": "Onshore and offshore turbine aerodynamics, Betz limit physics, drivetrains, and wake optimization.",
        "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/></svg>',
    },
    {
        "name": "Hydroelectric Energy",
        "desc": "Impoundment, run-of-river, Pelton/Francis/Kaplan turbines, and closed-loop pumped storage plants.",
        "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>',
    },
    {
        "name": "Biomass Energy",
        "desc": "Anaerobic digestion, biogas upgrading, second-generation biofuels, biochar, and BECCS pathways.",
        "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    },
    {
        "name": "Geothermal Energy",
        "desc": "Dry steam, flash steam, binary-cycle ORC power plants, enhanced geothermal systems, and heat pumps.",
        "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12h20"/><path d="M2 17h20"/><path d="M7 7l5-4 5 4"/><path d="M12 3v9"/></svg>',
    },
    {
        "name": "Green Hydrogen",
        "desc": "Alkaline, PEM, and SOEC water electrolysis, salt cavern storage, green ammonia, and clean steel.",
        "icon": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="7" cy="12" r="3"/><circle cx="17" cy="12" r="3"/><path d="M10 12h4"/><path d="M7 9V5"/><path d="M17 19v-4"/></svg>',
    },
]

EXAMPLE_QUESTIONS: list[tuple[str, str]] = [
    ("Solar Efficiency", "Explain solar panel efficiency across monocrystalline, polycrystalline, and perovskite cells."),
    ("Wind Generation", "How do wind turbines generate electricity and what is the Betz limit?"),
    ("Biomass Advantages", "What are the advantages of biomass energy and BECCS?"),
    ("Storage Challenges", "What are the challenges of renewable energy storage?"),
    ("Green Hydrogen", "Compare Alkaline, PEM, and SOEC electrolyzers for green hydrogen production."),
    ("Clean Energy Policy", "Explain the subsidy structure under PM Surya Ghar: Muft Bijli Yojana."),
]

HERO_ILLUSTRATION_SVG = """
<div class="hero-visual-frame">
<svg viewBox="0 0 460 310" width="100%" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Clean energy landscape with wind turbines and solar panels">
  <rect width="460" height="310" rx="8" fill="#141C19"/>
  <circle cx="358" cy="76" r="28" fill="#C9B37E" opacity="0.72"/>
  <circle cx="358" cy="76" r="42" fill="#C9B37E" opacity="0.14"/>
  <line x1="28" y1="64" x2="210" y2="64" stroke="#222E29" stroke-width="1"/>
  <line x1="28" y1="104" x2="170" y2="104" stroke="#222E29" stroke-width="1"/>
  <path d="M0 220 Q115 170 245 212 T460 198 L460 310 L0 310 Z" fill="#1C2823"/>
  <path d="M0 248 Q160 205 320 238 T460 228 L460 310 L0 310 Z" fill="#23342C"/>
  <path d="M0 272 Q190 250 460 268 L460 310 L0 310 Z" fill="#2B4036"/>
  <line x1="112" y1="98" x2="108" y2="238" stroke="#95D5B2" stroke-width="3" stroke-linecap="round"/>
  <circle cx="112" cy="98" r="4" fill="#95D5B2"/>
  <path d="M112 98 L112 46" stroke="#95D5B2" stroke-width="2.4" stroke-linecap="round"/>
  <path d="M112 98 L156 124" stroke="#95D5B2" stroke-width="2.4" stroke-linecap="round"/>
  <path d="M112 98 L68 124" stroke="#95D5B2" stroke-width="2.4" stroke-linecap="round"/>
  <line x1="214" y1="126" x2="211" y2="232" stroke="#74C69D" stroke-width="2.4" stroke-linecap="round"/>
  <circle cx="214" cy="126" r="3.2" fill="#74C69D"/>
  <path d="M214 126 L214 84" stroke="#74C69D" stroke-width="2" stroke-linecap="round"/>
  <path d="M214 126 L249 147" stroke="#74C69D" stroke-width="2" stroke-linecap="round"/>
  <path d="M214 126 L179 147" stroke="#74C69D" stroke-width="2" stroke-linecap="round"/>
  <line x1="292" y1="146" x2="290" y2="224" stroke="#52B788" stroke-width="2" stroke-linecap="round"/>
  <circle cx="292" cy="146" r="2.5" fill="#52B788"/>
  <path d="M292 146 L292 114" stroke="#52B788" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M292 146 L319 162" stroke="#52B788" stroke-width="1.6" stroke-linecap="round"/>
  <path d="M292 146 L265 162" stroke="#52B788" stroke-width="1.6" stroke-linecap="round"/>
  <line x1="242" y1="266" x2="242" y2="284" stroke="#8CA096" stroke-width="2.2"/>
  <line x1="308" y1="266" x2="308" y2="284" stroke="#8CA096" stroke-width="2.2"/>
  <polygon points="224,268 312,268 328,236 240,236" fill="#1B4332" stroke="#74C69D" stroke-width="1.1"/>
  <line x1="262" y1="236" x2="246" y2="268" stroke="#52B788" stroke-width="0.9"/>
  <line x1="284" y1="236" x2="268" y2="268" stroke="#52B788" stroke-width="0.9"/>
  <line x1="306" y1="236" x2="290" y2="268" stroke="#52B788" stroke-width="0.9"/>
  <line x1="232" y1="252" x2="320" y2="252" stroke="#52B788" stroke-width="0.9"/>
  <line x1="354" y1="266" x2="354" y2="284" stroke="#8CA096" stroke-width="2.2"/>
  <line x1="418" y1="266" x2="418" y2="284" stroke="#8CA096" stroke-width="2.2"/>
  <polygon points="336,268 424,268 440,236 352,236" fill="#2D6A4F" stroke="#95D5B2" stroke-width="1.1"/>
  <line x1="374" y1="236" x2="358" y2="268" stroke="#74C69D" stroke-width="0.9"/>
  <line x1="396" y1="236" x2="380" y2="268" stroke="#74C69D" stroke-width="0.9"/>
  <line x1="418" y1="236" x2="402" y2="268" stroke="#74C69D" stroke-width="0.9"/>
  <line x1="344" y1="252" x2="432" y2="252" stroke="#74C69D" stroke-width="0.9"/>
</svg>
</div>
"""

BRAND_LOGO_SVG = (
    '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="28" height="28" rx="6" fill="#2D6A4F"/>'
    '<path d="M19.5 8.5C13.5 8.5 9 12.2 9 17.5C9 18.4 9.15 19.2 9.4 19.9C14.7 19.9 19.5 15.8 19.5 8.5Z" '
    'stroke="#E6ECE8" stroke-width="1.75" stroke-linejoin="round"/>'
    '<path d="M8.5 20.5C10.5 17.5 13 15.3 16.5 13.5" stroke="#B7E4C7" stroke-width="1.6" stroke-linecap="round"/>'
    '</svg>'
)


def inject_custom_css() -> None:
    """Apply a natural, minimal dark-charcoal and forest-green stylesheet."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --forest: #2D6A4F;
            --forest-hover: #245740;
            --sage: #74C69D;
            --sage-tint: #1B2B23;
            --bg-dark: #0F1412;
            --surface-card: #161D1A;
            --surface-soft: #1C2521;
            --border-hairline: #27332D;
            --text-primary: #E6ECE8;
            --text-muted: #9AA7A0;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: var(--text-primary);
        }

        .stApp {
            background-color: var(--bg-dark);
        }

        h1, h2, h3, .display-font {
            font-family: 'DM Sans', 'Inter', sans-serif !important;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #131916;
            border-right: 1px solid #27332D;
        }

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 6px;
            font-weight: 500;
            font-size: 0.92rem;
            padding: 0.5rem 1.05rem;
            transition: background-color 0.16s ease, border-color 0.16s ease, color 0.16s ease;
        }

        div.stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background-color: #2D6A4F;
            border: 1px solid #2D6A4F;
            color: #F4F7F5;
        }

        div.stButton > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button[kind="primary"]:hover {
            background-color: #245740;
            border-color: #367D5E;
            color: #FFFFFF;
        }

        div.stButton > button[kind="secondary"] {
            background-color: #161D1A;
            border: 1px solid #2B3832;
            color: #E6ECE8;
        }

        div.stButton > button[kind="secondary"]:hover {
            background-color: #1D2923;
            border-color: #40916C;
            color: #FFFFFF;
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.15rem 0;
        }
        .nav-brand-text {
            display: flex;
            flex-direction: column;
            line-height: 1.2;
        }
        .nav-brand-title {
            font-family: 'DM Sans', 'Inter', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            color: #E6ECE8;
        }
        .nav-brand-sub {
            font-size: 0.76rem;
            color: #9AA7A0;
            font-weight: 500;
        }
        .nav-divider {
            border: 0;
            border-top: 1px solid #27332D;
            margin: 0.85rem 0 2rem 0;
        }

        .hero-eyebrow {
            font-size: 0.76rem;
            font-weight: 600;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: #74C69D;
            margin-bottom: 0.75rem;
        }
        .hero-heading {
            font-family: 'DM Sans', 'Inter', sans-serif;
            font-size: clamp(2rem, 3.6vw, 2.75rem);
            font-weight: 700;
            line-height: 1.14;
            color: #F2F5F3;
            margin: 0 0 1rem 0;
            letter-spacing: -0.025em;
        }
        .hero-subtitle {
            font-size: 1.06rem;
            line-height: 1.65;
            color: #A6B3AC;
            margin: 0 0 1.6rem 0;
            max-width: 560px;
        }
        .hero-visual-frame {
            border: 1px solid #27332D;
            border-radius: 8px;
            overflow: hidden;
            background: #141C19;
        }

        .stats-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            background: #161D1A;
            border: 1px solid #27332D;
            border-radius: 8px;
            margin: 2rem 0 2.5rem 0;
        }
        .stats-cell {
            padding: 1.15rem 1.35rem;
            border-right: 1px solid #27332D;
        }
        .stats-cell:last-child {
            border-right: none;
        }
        .stats-label {
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9AA7A0;
            margin-bottom: 0.3rem;
        }
        .stats-value {
            font-family: 'DM Sans', 'Inter', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: #F2F5F3;
        }
        .stats-note {
            font-size: 0.78rem;
            color: #7D8B84;
            margin-top: 0.15rem;
        }

        @media (max-width: 768px) {
            .stats-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .stats-cell {
                border-bottom: 1px solid #27332D;
            }
        }

        .section-kicker {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #74C69D;
            margin-bottom: 0.3rem;
        }
        .section-title {
            font-family: 'DM Sans', 'Inter', sans-serif;
            font-size: 1.55rem;
            font-weight: 700;
            color: #F2F5F3;
            margin: 0 0 0.45rem 0;
        }
        .section-lead {
            font-size: 0.96rem;
            color: #9AA7A0;
            margin: 0 0 1.4rem 0;
            line-height: 1.6;
        }

        .knowledge-card {
            background: #161D1A;
            border: 1px solid #27332D;
            border-radius: 8px;
            padding: 1.25rem 1.3rem;
            margin-bottom: 1rem;
            min-height: 162px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: border-color 0.18s ease, background-color 0.18s ease;
        }
        .knowledge-card:hover {
            border-color: #40916C;
            background: #1A231F;
        }
        .knowledge-card-head {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin-bottom: 0.6rem;
        }
        .knowledge-icon-wrap {
            width: 34px;
            height: 34px;
            border-radius: 6px;
            background: #1D2C25;
            border: 1px solid #2A3F34;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }
        .knowledge-card-title {
            font-family: 'DM Sans', 'Inter', sans-serif;
            font-size: 1.02rem;
            font-weight: 600;
            color: #F2F5F3;
        }
        .knowledge-card-desc {
            font-size: 0.88rem;
            line-height: 1.55;
            color: #A2AFA8;
            margin-bottom: 0.85rem;
        }
        .knowledge-card-meta {
            font-size: 0.78rem;
            font-weight: 500;
            color: #74C69D;
        }

        .chat-msg-user {
            background: #19221E;
            border: 1px solid #27332D;
            border-radius: 8px;
            padding: 0.95rem 1.15rem;
            margin-bottom: 0.75rem;
            color: #E6ECE8;
        }
        .chat-msg-assistant {
            background: #161D1A;
            border: 1px solid #27332D;
            border-left: 3px solid #40916C;
            border-radius: 8px;
            padding: 1.1rem 1.25rem;
            margin-bottom: 0.5rem;
            line-height: 1.65;
            color: #E6ECE8;
        }
        .chat-role-label {
            font-size: 0.73rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9AA7A0;
            margin-bottom: 0.35rem;
        }
        .chat-role-label.assistant-label {
            color: #74C69D;
        }

        .source-snippet-box {
            background: #111714;
            border: 1px solid #27332D;
            border-left: 3px solid #40916C;
            padding: 0.8rem 1rem;
            border-radius: 6px;
            font-size: 0.87rem;
            line-height: 1.55;
            color: #D2DDD7;
            margin-top: 0.35rem;
            margin-bottom: 0.75rem;
            white-space: pre-wrap;
        }

        .doc-meta-line {
            font-size: 0.82rem;
            color: #9AA7A0;
        }
        .doc-tag {
            display: inline-block;
            background: #1D2C25;
            color: #74C69D;
            border: 1px solid #2B4236;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 0.12rem 0.48rem;
            border-radius: 4px;
            margin-right: 0.45rem;
        }

        .rag-step {
            background: #161D1A;
            border: 1px solid #27332D;
            border-radius: 8px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.75rem;
        }
        .rag-step-num {
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #74C69D;
            margin-bottom: 0.2rem;
        }
        .rag-step-title {
            font-family: 'DM Sans', 'Inter', sans-serif;
            font-size: 0.98rem;
            font-weight: 600;
            color: #F2F5F3;
            margin-bottom: 0.25rem;
        }
        .rag-step-body {
            font-size: 0.88rem;
            color: #A2AFA8;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_manager(model_name: str = DEFAULT_EMBEDDING_MODEL) -> EmbeddingManager:
    """Cache the embedding model across Streamlit reruns."""
    return EmbeddingManager(model_name=model_name)


@st.cache_resource(show_spinner="Connecting to vector database...")
def get_vector_database() -> VectorDatabase:
    """Cache the persistent ChromaDB client across reruns."""
    return VectorDatabase(persist_dir=CHROMA_DIR)


def init_session_state() -> None:
    """Initialize required Streamlit session state keys."""
    defaults: dict[str, Any] = {
        "nav_page": "Home",
        "chat_history": [],
        "gemini_model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        "pending_prompt": None,
        "upload_results_feedback": None,
        "sample_sync_feedback": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.nav_page not in NAVIGATION_PAGES:
        st.session_state.nav_page = "Home"


def navigate_to(page: str) -> None:
    """Switch the active page cleanly via button callbacks."""
    if page in NAVIGATION_PAGES:
        st.session_state.nav_page = page


def index_file_to_kb(
    file_path: Path,
    processor: DocumentProcessor,
    embedder: EmbeddingManager,
    vector_db: VectorDatabase,
    category: str = "Auto-Detect",
    raw_bytes: bytes | None = None,
) -> tuple[str, str]:
    """Process, embed, and store a single document in ChromaDB with duplicate checks."""
    try:
        content_bytes = raw_bytes if raw_bytes is not None else file_path.read_bytes()
        file_hash = processor.compute_file_hash(content_bytes)
        exists, reason = vector_db.document_exists(file_path.name, file_hash)
        if exists:
            return "skipped", f"{file_path.name}: {reason}"

        processed = processor.process_file(
            file_path=file_path,
            category=category,
            raw_bytes=content_bytes,
        )
        texts = [chunk.text for chunk in processed.chunks]
        embeddings = embedder.embed_documents(texts)
        added_count = vector_db.add_document_chunks(processed.chunks, embeddings)
        return (
            "indexed",
            f"{processed.filename} ({processed.category}, {processed.page_count} page(s), {added_count} chunks)",
        )
    except PDFProcessingError as exc:
        return "error", f"{file_path.name} [PDF Processing]: {exc}"
    except TextEncodingError as exc:
        return "error", f"{file_path.name} [Text Encoding]: {exc}"
    except Exception as exc:
        return "error", f"{file_path.name}: {exc}"


def index_sample_dataset(
    embedder: EmbeddingManager,
    vector_db: VectorDatabase,
) -> tuple[int, int, list[str]]:
    """Index all sample PDF/TXT files located in data/documents/."""
    processor = DocumentProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", 700)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100)),
    )
    sample_files = processor.list_disk_documents(DATA_DIR)
    if not sample_files:
        return 0, 0, ["No sample documents found in data/documents/."]

    indexed_count = 0
    skipped_count = 0
    messages: list[str] = []
    for path in sample_files:
        status, msg = index_file_to_kb(path, processor, embedder, vector_db, category="Auto-Detect")
        if status == "indexed":
            indexed_count += 1
        elif status == "skipped":
            skipped_count += 1
        messages.append(f"[{status.upper()}] {msg}")
    return indexed_count, skipped_count, messages


def render_top_navbar() -> str:
    """Render the minimal horizontal website navigation bar."""
    active_page = st.session_state.nav_page
    brand_col, nav_col = st.columns([1.45, 1.55], vertical_alignment="center")

    with brand_col:
        st.markdown(
            f"""
            <div class="nav-brand">
                {BRAND_LOGO_SVG}
                <div class="nav-brand-text">
                    <span class="nav-brand-title">Renewable Energy Knowledge Assistant</span>
                    <span class="nav-brand-sub">Clean Energy Research &amp; RAG Platform</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col:
        btn_cols = st.columns(len(NAVIGATION_PAGES))
        for idx, page_name in enumerate(NAVIGATION_PAGES):
            btn_type = "primary" if active_page == page_name else "secondary"
            with btn_cols[idx]:
                st.button(
                    page_name,
                    key=f"top_nav_{page_name}",
                    type=btn_type,
                    width="stretch",
                    on_click=navigate_to,
                    args=(page_name,),
                )

    st.markdown('<hr class="nav-divider">', unsafe_allow_html=True)
    return st.session_state.nav_page


def render_sidebar_settings(vector_db: VectorDatabase, rag_pipeline: RAGPipeline) -> None:
    """Render minimal collapsible system settings in the sidebar."""
    with st.sidebar:
        st.markdown("### System Configuration")
        stats = vector_db.get_stats()
        c1, c2 = st.columns(2)
        c1.metric("Documents", stats["total_documents"])
        c2.metric("Chunks", stats["total_chunks"])

        st.divider()
        if rag_pipeline.is_api_key_configured():
            st.success("Gemini API Key Active")
        else:
            st.warning("Gemini API Key Not Set")

        model_options = list(MODEL_OPTIONS)
        current_model = st.session_state.gemini_model
        if current_model not in model_options:
            model_options.insert(0, current_model)
        st.session_state.gemini_model = st.selectbox(
            "Gemini Model",
            options=model_options,
            index=model_options.index(current_model),
        )

        st.divider()
        if st.button("Clear Conversation", width="stretch"):
            st.session_state.chat_history = []
            st.toast("Conversation cleared.")


def render_home(embedder: EmbeddingManager, vector_db: VectorDatabase) -> None:
    """Render Home: Hero section, repository summary strip, and Knowledge Overview."""
    hero_left, hero_right = st.columns([1.28, 1.0], gap="large", vertical_alignment="center")

    with hero_left:
        st.markdown(
            """
            <div class="hero-eyebrow">Clean Energy Intelligence</div>
            <h1 class="hero-heading">Explore the Future of Renewable Energy</h1>
            <p class="hero-subtitle">
                Your intelligent knowledge companion for understanding clean energy,
                sustainable technologies, and renewable power solutions.
            </p>
            """,
            unsafe_allow_html=True,
        )
        cta1, cta2, _ = st.columns([1.05, 1.0, 0.7])
        with cta1:
            st.button(
                "Explore Knowledge",
                type="primary",
                width="stretch",
                on_click=navigate_to,
                args=("Knowledge Base",),
            )
        with cta2:
            st.button(
                "Ask Assistant",
                type="secondary",
                width="stretch",
                on_click=navigate_to,
                args=("Ask Assistant",),
            )

    with hero_right:
        st.markdown(HERO_ILLUSTRATION_SVG, unsafe_allow_html=True)

    stats = vector_db.get_stats()
    model_info = embedder.get_model_info()
    st.markdown(
        f"""
        <div class="stats-strip">
            <div class="stats-cell">
                <div class="stats-label">Indexed Documents</div>
                <div class="stats-value">{stats['total_documents']}</div>
                <div class="stats-note">PDF and TXT literature</div>
            </div>
            <div class="stats-cell">
                <div class="stats-label">Indexed Chunks</div>
                <div class="stats-value">{stats['total_chunks']}</div>
                <div class="stats-note">Semantic vector passages</div>
            </div>
            <div class="stats-cell">
                <div class="stats-label">Knowledge Domains</div>
                <div class="stats-value">8</div>
                <div class="stats-note">Specialized energy sectors</div>
            </div>
            <div class="stats-cell">
                <div class="stats-label">Embedding Space</div>
                <div class="stats-value">{model_info['embedding_dimension']}d</div>
                <div class="stats-note">{html.escape(str(model_info['model_name']))}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-kicker">Knowledge Overview</div>
        <h2 class="section-title">Core Renewable Energy Domains</h2>
        <p class="section-lead">
            Browse curated technical documentation across major clean-energy generation, conversion,
            and decarbonization technologies.
        </p>
        """,
        unsafe_allow_html=True,
    )

    breakdown = stats["categories_breakdown"]
    row_top = st.columns(3, gap="medium")
    row_bottom = st.columns(3, gap="medium")
    grid_cols = row_top + row_bottom

    for idx, domain in enumerate(CORE_OVERVIEW_DOMAINS):
        doc_count = breakdown.get(domain["name"], 0)
        with grid_cols[idx]:
            st.markdown(
                f"""
                <div class="knowledge-card">
                    <div>
                        <div class="knowledge-card-head">
                            <span class="knowledge-icon-wrap">{domain['icon']}</span>
                            <span class="knowledge-card-title">{domain['name']}</span>
                        </div>
                        <div class="knowledge-card-desc">{domain['desc']}</div>
                    </div>
                    <div class="knowledge-card-meta">{doc_count} document(s) indexed</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    storage_count = breakdown.get("Renewable Energy Storage", 0)
    policy_count = breakdown.get("Government Renewable Energy Schemes", 0)
    st.caption(
        f"Also indexed in Knowledge Base: Renewable Energy Storage ({storage_count} document(s)) "
        f"and Government Renewable Energy Schemes ({policy_count} document(s))."
    )


def render_knowledge_base(embedder: EmbeddingManager, vector_db: VectorDatabase) -> None:
    """Render Knowledge Base: Document upload, repository stats, and document management."""
    st.markdown(
        """
        <div class="section-kicker">Document Repository</div>
        <h1 class="section-title" style="font-size: 1.9rem;">Knowledge Base &amp; Document Management</h1>
        <p class="section-lead">
            Upload PDF or TXT files to expand the vector index incrementally, inspect indexed categories,
            or remove outdated documents.
        </p>
        """,
        unsafe_allow_html=True,
    )

    stats = vector_db.get_stats()
    docs = vector_db.list_documents()

    m1, m2, m3 = st.columns(3)
    m1.metric("Indexed Documents", stats["total_documents"])
    m2.metric("Total Indexed Chunks", stats["total_chunks"])
    m3.metric("Active Categories", len(stats["categories_breakdown"]))

    st.markdown("")
    with st.container(border=True):
        st.markdown("#### Upload Documents")
        st.caption("Supported formats: PDF and TXT. Duplicate files are automatically skipped.")

        up_col, cat_col = st.columns([2, 1])
        with up_col:
            uploaded_files = st.file_uploader(
                "Select PDF or TXT documents",
                type=["pdf", "txt"],
                accept_multiple_files=True,
            )
        with cat_col:
            selected_category = st.selectbox(
                "Knowledge Category",
                options=["Auto-Detect"] + RENEWABLE_CATEGORIES,
                help="Select a specific category or auto-detect per page from text.",
            )

        with st.expander("Chunking Settings", expanded=False):
            c1, c2 = st.columns(2)
            chunk_size = c1.slider(
                "Chunk Size (characters)",
                min_value=300,
                max_value=1500,
                value=int(os.getenv("CHUNK_SIZE", 700)),
                step=50,
            )
            chunk_overlap = c2.slider(
                "Chunk Overlap (characters)",
                min_value=0,
                max_value=300,
                value=int(os.getenv("CHUNK_OVERLAP", 100)),
                step=25,
            )

        btn_col1, btn_col2 = st.columns([1.2, 1])
        with btn_col1:
            process_clicked = st.button(
                "Process & Index Uploaded Files",
                type="primary",
                disabled=not uploaded_files,
                width="stretch",
            )
        with btn_col2:
            sample_files = DocumentProcessor.list_disk_documents(DATA_DIR)
            sync_clicked = st.button(
                f"Sync Local Dataset ({len(sample_files)} files)",
                type="secondary",
                disabled=len(sample_files) == 0,
                width="stretch",
            )

        if process_clicked:
            if chunk_overlap >= chunk_size:
                st.error("Chunk overlap must be smaller than chunk size.")
                return

            processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            progress_bar = st.progress(0.0, text="Processing documents...")
            results_summary: list[tuple[str, str]] = []
            total_files = len(uploaded_files)

            for idx, uploaded in enumerate(uploaded_files, start=1):
                progress_bar.progress(
                    (idx - 1) / total_files,
                    text=f"Processing {uploaded.name} ({idx}/{total_files})...",
                )
                raw_bytes = uploaded.getvalue()
                file_hash = processor.compute_file_hash(raw_bytes)
                safe_name = processor.sanitize_filename(uploaded.name)

                exists, reason = vector_db.document_exists(safe_name, file_hash)
                if exists:
                    results_summary.append(("skipped", f"{safe_name}: {reason}"))
                    continue

                saved_path = processor.save_uploaded_bytes(safe_name, raw_bytes, DATA_DIR)
                status, message = index_file_to_kb(
                    file_path=saved_path,
                    processor=processor,
                    embedder=embedder,
                    vector_db=vector_db,
                    category=selected_category,
                    raw_bytes=raw_bytes,
                )
                if status == "error" and saved_path.exists():
                    saved_path.unlink(missing_ok=True)
                results_summary.append((status, message))

            progress_bar.progress(1.0, text="Processing complete.")
            st.session_state.upload_results_feedback = results_summary
            st.rerun()

        if sync_clicked:
            with st.spinner("Checking and indexing local documents..."):
                indexed, skipped, logs = index_sample_dataset(embedder, vector_db)
            st.session_state.sample_sync_feedback = (indexed, skipped, logs)
            st.rerun()

        if st.session_state.upload_results_feedback:
            for status, message in st.session_state.upload_results_feedback:
                if status == "indexed":
                    st.success(f"Indexed: {message}")
                elif status == "skipped":
                    st.info(f"Skipped duplicate: {message}")
                else:
                    st.error(f"Failed: {message}")
            st.session_state.upload_results_feedback = None

        if st.session_state.sample_sync_feedback:
            indexed, skipped, logs = st.session_state.sample_sync_feedback
            st.success(f"Sync complete: {indexed} newly indexed, {skipped} already present.")
            with st.expander("View Sync Log"):
                for line in logs:
                    st.write(line)
            st.session_state.sample_sync_feedback = None

    st.markdown("")
    st.markdown("#### Uploaded Document List")

    if docs:
        all_doc_categories = sorted({cat for d in docs for cat in d.get("categories", [d["category"]])})
        filter_col, search_col = st.columns([1, 2])
        cat_filter = filter_col.selectbox(
            "Filter by Category",
            options=["All Categories"] + all_doc_categories,
        )
        search_term = search_col.text_input("Search by filename", placeholder="e.g., solar, hydrogen, turbine...")

        filtered_docs = [
            d
            for d in docs
            if (cat_filter == "All Categories" or cat_filter in d.get("categories", [d["category"]]))
            and (not search_term or search_term.lower() in d["filename"].lower())
        ]

        if not filtered_docs:
            st.info("No documents match the current filter.")
        else:
            for doc in filtered_docs:
                with st.container(border=True):
                    c_info, c_action = st.columns([5.2, 1], vertical_alignment="center")
                    with c_info:
                        safe_name = html.escape(doc["filename"])
                        safe_type = html.escape(doc["file_type"])
                        safe_cat = html.escape(doc["category"])
                        safe_time = html.escape(doc["uploaded_at"])
                        st.markdown(
                            f'<span class="doc-tag">{safe_type}</span><strong>{safe_name}</strong><br>'
                            f'<span class="doc-meta-line">Category: {safe_cat} &middot; '
                            f'Pages: {doc["page_count"]} &middot; Chunks: {doc["chunk_count"]} &middot; '
                            f'Indexed: {safe_time}</span>',
                            unsafe_allow_html=True,
                        )
                    with c_action:
                        if st.button("Delete", key=f"del_{doc['filename']}", width="stretch"):
                            removed_chunks = vector_db.delete_document(doc["filename"])
                            disk_file = DATA_DIR / doc["filename"]
                            if disk_file.exists():
                                disk_file.unlink(missing_ok=True)
                            st.success(f"Removed `{doc['filename']}` ({removed_chunks} chunk(s)).")
                            st.rerun()

        with st.expander("Category Breakdown", expanded=False):
            cat_rows: dict[str, dict[str, int]] = {}
            for doc in docs:
                for cat, ch_cnt in doc.get("category_chunks", {doc["category"]: doc["chunk_count"]}).items():
                    if cat not in cat_rows:
                        cat_rows[cat] = {"Documents": 0, "Chunks": 0}
                    cat_rows[cat]["Documents"] += 1
                    cat_rows[cat]["Chunks"] += ch_cnt
            cat_summary = pd.DataFrame(
                [
                    {"Category": cat, "Documents": vals["Documents"], "Chunks": vals["Chunks"]}
                    for cat, vals in cat_rows.items()
                ]
            ).sort_values("Chunks", ascending=False)
            st.dataframe(cat_summary, width="stretch", hide_index=True)

        with st.expander("Reset Knowledge Base", expanded=False):
            st.warning(
                "Clearing the knowledge base permanently deletes all indexed vectors from ChromaDB "
                "and removes stored files in `data/documents/`."
            )
            if st.button("Confirm Clear Entire Knowledge Base", type="secondary"):
                vector_db.clear_all()
                for disk_file in DocumentProcessor.list_disk_documents(DATA_DIR):
                    disk_file.unlink(missing_ok=True)
                st.success("Knowledge base cleared.")
                st.rerun()
    else:
        st.info("No documents are currently indexed in the knowledge base.")


def _render_sources_expander(sources: list[dict[str, Any]]) -> None:
    """Render expandable source references beneath grounded assistant answers."""
    if not sources:
        return
    with st.expander(f"Source References ({len(sources)} excerpt(s))", expanded=False):
        for idx, src in enumerate(sources, start=1):
            st.markdown(
                f"**Source {idx}:** `{src['filename']}` &nbsp;|&nbsp; "
                f"**Page:** `{src['page_number']}` &nbsp;|&nbsp; "
                f"**Category:** `{src['category']}` &nbsp;|&nbsp; "
                f"**Relevance:** `{src['similarity_score']:.1%}`"
            )
            safe_snippet = html.escape(src["snippet"])
            st.markdown(
                f'<div class="source-snippet-box">{safe_snippet}</div>',
                unsafe_allow_html=True,
            )


def render_ask_assistant(rag_pipeline: RAGPipeline, vector_db: VectorDatabase) -> None:
    """Render Ask Assistant: clean research-assistant conversational workspace."""
    st.markdown(
        """
        <div class="section-kicker">Research Assistant</div>
        <h1 class="section-title" style="font-size: 1.9rem;">Ask Renewable Energy Assistant</h1>
        <p class="section-lead">
            Ask technical or policy questions grounded strictly in your indexed renewable energy documents,
            complete with page-level source verification.
        </p>
        """,
        unsafe_allow_html=True,
    )

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2.0, 1.0, 1.3, 1.0], vertical_alignment="bottom")
    with ctrl1:
        category_filter = st.selectbox(
            "Domain Filter",
            options=["All Categories"] + RENEWABLE_CATEGORIES,
        )
    with ctrl2:
        top_k = st.slider(
            "Top-K Chunks",
            min_value=2,
            max_value=8,
            value=int(os.getenv("TOP_K_RESULTS", 4)),
        )
    with ctrl3:
        model_options = list(MODEL_OPTIONS)
        current_model = st.session_state.gemini_model
        if current_model not in model_options:
            model_options.insert(0, current_model)
        st.session_state.gemini_model = st.selectbox(
            "Model",
            options=model_options,
            index=model_options.index(current_model),
            key="ask_model_select",
        )
    with ctrl4:
        if st.button("Clear Conversation", width="stretch"):
            st.session_state.chat_history = []
            st.rerun()

    stats = vector_db.get_stats()
    if stats["total_chunks"] == 0:
        st.warning(
            "Your knowledge base currently has 0 indexed chunks. Open **Knowledge Base** to upload "
            "documents or sync the local dataset before asking questions."
        )

    with st.expander("Suggested Research Questions", expanded=len(st.session_state.chat_history) == 0):
        btn_cols = st.columns(3)
        for idx, (label, question_text) in enumerate(EXAMPLE_QUESTIONS):
            with btn_cols[idx % 3]:
                if st.button(f"{label}", key=f"ex_q_{idx}", width="stretch", help=question_text):
                    st.session_state.pending_prompt = question_text

    # Render conversation history cleanly
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            safe_q = html.escape(message["content"])
            st.markdown(
                f"""
                <div class="chat-msg-user">
                    <div class="chat-role-label">Question</div>
                    <div>{safe_q}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            msg_status = message.get("status", "answered")
            if msg_status == "out_of_domain":
                st.warning(message["content"])
            elif msg_status == "unavailable":
                st.info(message["content"])
            else:
                st.markdown(
                    '<div class="chat-msg-assistant"><div class="chat-role-label assistant-label">Grounded Answer</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(message["content"])
                st.markdown("</div>", unsafe_allow_html=True)

            if message.get("sources"):
                _render_sources_expander(message["sources"])

    # Large question input field with dedicated Send button
    with st.form("ask_assistant_form", clear_on_submit=True):
        typed_query = st.text_area(
            "Your Question",
            height=96,
            placeholder="Ask a question grounded in your renewable energy documents (e.g., What are the challenges of renewable energy storage?)...",
            label_visibility="collapsed",
        )
        _, send_col = st.columns([4.5, 1])
        with send_col:
            submitted = st.form_submit_button("Send", type="primary", width="stretch")

    user_query = ""
    if st.session_state.pending_prompt:
        user_query = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    elif submitted and typed_query.strip():
        user_query = typed_query.strip()

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("Searching indexed knowledge and generating grounded response..."):
            rag_result = rag_pipeline.answer_question(
                question=user_query,
                category_filter=category_filter,
                top_k=top_k,
                chat_history=st.session_state.chat_history[:-1],
            )

        if not rag_result.domain_relevant:
            msg_status = "out_of_domain"
        elif not rag_result.found_in_context:
            msg_status = "unavailable"
        else:
            msg_status = "answered"

        serialized_sources = (
            [
                {
                    "filename": s.filename,
                    "page_number": s.page_number,
                    "chunk_index": s.chunk_index,
                    "category": s.category,
                    "similarity_score": s.similarity_score,
                    "snippet": s.snippet,
                }
                for s in rag_result.sources
            ]
            if rag_result.found_in_context
            else []
        )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": rag_result.answer,
                "status": msg_status,
                "sources": serialized_sources,
            }
        )
        st.rerun()


def render_about() -> None:
    """Render About: Project purpose, RAG architecture, and technology stack."""
    st.markdown(
        """
        <div class="section-kicker">Architecture &amp; Purpose</div>
        <h1 class="section-title" style="font-size: 1.9rem;">About Renewable Energy Knowledge Assistant</h1>
        <p class="section-lead">
            Built to help students, engineers, and clean-energy researchers extract accurate,
            source-verified insights directly from renewable energy technical literature and policy documents.
        </p>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.2, 1], gap="large")

    with left_col:
        st.markdown("#### How Retrieval-Augmented Generation (RAG) Works")
        steps = [
            (
                "Step 01",
                "Document Ingestion & Cleaning",
                "Uploaded PDF and TXT files are extracted page-by-page using PyPDF, normalized in UTF-8, "
                "and cleaned while preserving domain symbols and metadata (filename and page number).",
            ),
            (
                "Step 02",
                "Sentence-Aware Chunking & Embeddings",
                "Documents are split into overlapping 700-character segments (100-character overlap) and "
                "converted into dense normalized embeddings via Sentence Transformers (all-MiniLM-L6-v2).",
            ),
            (
                "Step 03",
                "Semantic Search & Reranking in ChromaDB",
                "User questions are validated for renewable energy relevance, embedded in the same vector space, "
                "filtered by cosine similarity threshold, and reranked for topic precision.",
            ),
            (
                "Step 04",
                "Strictly Grounded Gemini Synthesis & Citations",
                "Only verified high-relevance passages are passed to Google Gemini. Every claim is cited inline "
                "and paired with expandable source snippets, or cleanly declined when unavailable.",
            ),
        ]
        for num, title, body in steps:
            st.markdown(
                f"""
                <div class="rag-step">
                    <div class="rag-step-num">{num}</div>
                    <div class="rag-step-title">{title}</div>
                    <div class="rag-step-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("#### Technology Stack")
        with st.container(border=True):
            st.markdown(
                """
                - **Language:** Python 3.12
                - **Frontend Interface:** Streamlit
                - **LLM Reasoning:** Google Gemini API (`google-genai`)
                - **Embedding Model:** Sentence Transformers (`all-MiniLM-L6-v2`)
                - **Vector Store:** ChromaDB (Persistent Cosine Index)
                - **PDF Extraction:** PyPDF
                """
            )

        st.markdown("#### Supported Renewable Energy Domains")
        with st.container(border=True):
            for idx, category in enumerate(RENEWABLE_CATEGORIES, start=1):
                st.markdown(f"{idx}. **{category}**")


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="Renewable Energy Knowledge Assistant",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_custom_css()
    init_session_state()

    embedder = get_embedding_manager(os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    vector_db = get_vector_database()
    rag_pipeline = RAGPipeline(
        embedding_manager=embedder,
        vector_db=vector_db,
        model_name=st.session_state.gemini_model,
        top_k=int(os.getenv("TOP_K_RESULTS", 4)),
        min_similarity=float(os.getenv("MIN_SIMILARITY_SCORE", 0.32)),
    )

    render_sidebar_settings(vector_db, rag_pipeline)
    active_page = render_top_navbar()

    if active_page == "Home":
        render_home(embedder, vector_db)
    elif active_page == "Knowledge Base":
        render_knowledge_base(embedder, vector_db)
    elif active_page == "Ask Assistant":
        render_ask_assistant(rag_pipeline, vector_db)
    elif active_page == "About":
        render_about()


if __name__ == "__main__":
    main()
