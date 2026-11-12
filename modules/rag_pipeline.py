"""End-to-end Retrieval-Augmented Generation (RAG) pipeline backed by Gemini and ChromaDB."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from modules.document_processor import DocumentProcessor, TextEncodingError
from modules.embeddings import EmbeddingManager
from modules.vector_database import RetrievedChunk, VectorDatabase


class GeminiAPIError(RuntimeError):
    """Raised when the Google Gemini API fails to generate a response."""

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
FALLBACK_GEMINI_MODELS = ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash")
_PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "YOUR_API_KEY", "replace_me"}

UNAVAILABLE_MESSAGE = (
    "Sorry, the uploaded documents do not contain relevant information about this question."
)
OUT_OF_DOMAIN_MESSAGE = (
    "Sorry, I am an AI-Powered Renewable Energy Knowledge Assistant and can only answer "
    "questions related to renewable energy topics such as solar, wind, hydroelectric, "
    "geothermal, biomass, green hydrogen, energy storage, or government renewable energy schemes."
)

_DOMAIN_ANCHORS = (
    "Solar photovoltaic panels, wind turbines, hydroelectric power, geothermal energy, "
    "biomass biofuels, green hydrogen electrolysis, renewable battery energy storage, "
    "and government clean energy policies and subsidies.",
)

_SPECIFIC_DOMAIN_KEYWORDS = (
    "solar",
    "photovoltaic",
    "photovoltaics",
    "pv",
    "monocrystalline",
    "polycrystalline",
    "perovskite",
    "agrivoltaics",
    "csp",
    "concentrated solar",
    "parabolic trough",
    "heliostat",
    "inverter",
    "mppt",
    "wind",
    "turbine",
    "turbines",
    "offshore",
    "onshore",
    "betz",
    "nacelle",
    "rotor",
    "hydro",
    "hydropower",
    "hydroelectric",
    "pumped storage",
    "psh",
    "pelton",
    "francis",
    "kaplan",
    "penstock",
    "cavitation",
    "geothermal",
    "egs",
    "gshp",
    "heat pump",
    "dry steam",
    "flash steam",
    "binary cycle",
    "biomass",
    "biogas",
    "biofuel",
    "biofuels",
    "bioenergy",
    "bioethanol",
    "biodiesel",
    "biochar",
    "beccs",
    "anaerobic digestion",
    "gasification",
    "pyrolysis",
    "syngas",
    "saf",
    "hvo",
    "hydrogen",
    "green ammonia",
    "electrolyzer",
    "electrolyzers",
    "electrolyser",
    "electrolysis",
    "fuel cell",
    "pemfc",
    "sofc",
    "soec",
    "renewable",
    "renewables",
    "clean energy",
    "green energy",
    "green power",
    "energy storage",
    "battery",
    "batteries",
    "bess",
    "lfp",
    "nmc",
    "vanadium",
    "flow battery",
    "sodium-ion",
    "iron-air",
    "caes",
    "laes",
    "molten salt",
    "thermal storage",
    "surya ghar",
    "kusum",
    "nghm",
    "pli",
    "mnre",
    "net metering",
    "gross metering",
    "feed-in tariff",
    "cbam",
    "almm",
    "rpo",
    "rgo",
    "ists",
    "microgrid",
    "virtual power plant",
    "grid-forming",
    "decarbonization",
    "tidal",
    "wave energy",
    "otec",
    "ocean thermal",
)

_BROAD_ENERGY_KEYWORDS = ("energy", "power", "electricity", "grid", "panel", "panels", "tariff")

_FOLLOWUP_PRONOUNS = {
    "it",
    "its",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
    "above",
    "more",
}

_GENERIC_QUERY_WORDS = {
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "are",
    "is",
    "was",
    "were",
    "can",
    "could",
    "does",
    "did",
    "do",
    "explain",
    "describe",
    "define",
    "tell",
    "give",
    "list",
    "compare",
    "difference",
    "between",
    "about",
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "that",
    "this",
    "advantages",
    "disadvantages",
    "benefits",
    "challenges",
    "limitations",
    "uses",
    "applications",
    "working",
    "principle",
    "principles",
    "generate",
    "generates",
    "generating",
    "production",
    "produce",
    "electricity",
    "energy",
    "renewable",
    "power",
    "system",
    "systems",
}

SYSTEM_INSTRUCTION = """You are the AI-Powered Renewable Energy Knowledge Assistant.
Your strict responsibility is to answer the user's question ONLY using the provided Context Excerpts retrieved from uploaded renewable energy documents.

Rules you MUST follow:
1. Ground every factual claim strictly in the provided Context Excerpts. Never use outside knowledge, assumptions, or unstated facts.
2. When a question asks about advantages, benefits, challenges, limitations, or comparisons, synthesize ONLY the relevant properties, constraints, costs, risks, or capabilities explicitly stated in the Context Excerpts.
3. Cite sources inline for every factual statement using the exact format [Source: <filename>, Page <page_number>].
4. Never invent filenames, page numbers, statistics, or explanations that do not appear in the Context Excerpts.
5. If the provided Context Excerpts do not contain information relevant to answering the question (for example, if the excerpts discuss a different technology or topic than the one asked about), respond ONLY with this exact sentence and nothing else:
Sorry, the uploaded documents do not contain relevant information about this question."""


@dataclass
class SourceReference:
    """Structured citation metadata for UI display."""

    filename: str
    page_number: int
    chunk_index: int
    category: str
    similarity_score: float
    snippet: str


@dataclass
class RAGResponse:
    """Complete output returned by RAGPipeline.answer_question."""

    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    found_in_context: bool = False
    domain_relevant: bool = True
    model_used: str = DEFAULT_GEMINI_MODEL
    retrieval_count: int = 0
    error: str | None = None


class RAGPipeline:
    """Coordinates domain validation, query embedding, reranked retrieval, and grounded Gemini generation."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_db: VectorDatabase,
        api_key: str | None = None,
        model_name: str | None = None,
        top_k: int = 4,
        min_similarity: float = 0.32,
    ) -> None:
        load_dotenv(encoding="utf-8", override=False)
        self.embedding_manager = embedding_manager
        self.vector_db = vector_db
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.model_name = (model_name or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
        self._explicit_api_key = api_key.strip() if api_key else None
        self._domain_anchor_vec: np.ndarray | None = None

    @staticmethod
    def ensure_utf8_text(text: str) -> str:
        """Ensure valid UTF-8 Unicode text while preserving special symbols (– — ‘ ’ \" ° × ≥ ≤)."""
        return DocumentProcessor.ensure_utf8_text(text)

    def get_effective_api_key(self) -> str:
        """Resolve Gemini API key from explicit override or environment variable."""
        return self._explicit_api_key or os.getenv("GEMINI_API_KEY", "").strip()

    def is_api_key_configured(self) -> bool:
        """Return True when a non-placeholder Gemini API key is available."""
        key = self.get_effective_api_key()
        return bool(key) and key not in _PLACEHOLDER_KEYS

    def _get_domain_anchor_vector(self) -> np.ndarray:
        """Lazily compute and cache the normalized embedding of the renewable-energy domain anchor."""
        if self._domain_anchor_vec is None:
            vec = self.embedding_manager.embed_documents(_DOMAIN_ANCHORS)[0]
            self._domain_anchor_vec = np.array(vec, dtype=float)
        return self._domain_anchor_vec

    def is_renewable_energy_question(
        self,
        question: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """Validate whether the question belongs to the renewable energy domain.

        Returns (is_relevant, retrieval_query) where retrieval_query resolves short follow-up turns.
        """
        cleaned = (question or "").strip()
        lowered = cleaned.lower()

        # Check explicit specific renewable-energy vocabulary first
        if any(re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in _SPECIFIC_DOMAIN_KEYWORDS):
            return True, cleaned

        # Check conversational follow-ups when prior history exists
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        if chat_history and len(tokens) <= 12 and (tokens & _FOLLOWUP_PRONOUNS):
            last_user = next(
                (
                    str(m.get("content", "")).strip()
                    for m in reversed(chat_history)
                    if m.get("role") == "user" and str(m.get("content", "")).strip()
                ),
                "",
            )
            if last_user:
                combined = f"{last_user} {cleaned}"
                if any(
                    re.search(rf"\b{re.escape(kw)}\b", combined.lower())
                    for kw in _SPECIFIC_DOMAIN_KEYWORDS
                ):
                    return True, combined

        # Semantic domain similarity check using the same SentenceTransformer model
        try:
            q_vec = np.array(self.embedding_manager.embed_query(cleaned), dtype=float)
            domain_sim = float(np.dot(q_vec, self._get_domain_anchor_vector()))
        except Exception:
            domain_sim = 0.0

        has_broad_energy = any(
            re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in _BROAD_ENERGY_KEYWORDS
        )
        threshold = 0.18 if has_broad_energy else 0.24
        return (domain_sim >= threshold), cleaned

    @staticmethod
    def _extract_topic_terms(question: str) -> list[str]:
        """Extract specific topic terms (with simple stemming) from the user's question."""
        raw_tokens = re.findall(r"[a-z0-9]{3,}", question.lower())
        terms: list[str] = []
        for tok in raw_tokens:
            if tok in _GENERIC_QUERY_WORDS:
                continue
            # Preserve exact word or singular stem for clean matching
            stem = tok[:-1] if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss") else tok
            if stem not in terms:
                terms.append(stem)
        return terms

    @classmethod
    def _rerank_and_filter_chunks(
        cls,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks by semantic + topic-term coverage/density and prune tangential hits."""
        if not chunks:
            return []

        topic_terms = cls._extract_topic_terms(question)
        scored: list[tuple[float, float, RetrievedChunk]] = []

        for chunk in chunks:
            body_lower = chunk.text.lower()
            if topic_terms:
                matched_terms = 0
                density_sum = 0.0
                for term in topic_terms:
                    cnt = len(re.findall(rf"\b{re.escape(term)}[a-z]*\b", body_lower))
                    if cnt > 0:
                        matched_terms += 1
                        density_sum += min(cnt, 3) / 3.0
                coverage = matched_terms / len(topic_terms)
                density = density_sum / len(topic_terms)
            else:
                coverage = 0.0
                density = 0.0

            combined_score = chunk.similarity_score + (0.16 * coverage) + (0.09 * density)
            scored.append((combined_score, coverage, chunk))

        scored.sort(key=lambda item: (item[0], item[2].similarity_score), reverse=True)

        # If specific topic terms exist and chunks match them, discard zero-overlap tangential chunks
        if topic_terms and any(cov > 0.0 for _, cov, _ in scored):
            scored = [item for item in scored if item[1] > 0.0]

        best_combined = scored[0][0]
        best_sim = max(item[2].similarity_score for item in scored)

        filtered: list[RetrievedChunk] = []
        for combined_score, _, chunk in scored:
            if chunk.similarity_score < best_sim - 0.14:
                continue
            if combined_score < best_combined - 0.18:
                continue
            filtered.append(chunk)
            if len(filtered) >= top_k:
                break

        return filtered

    @classmethod
    def _build_context_block(cls, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks with UTF-8 source metadata for the Gemini prompt."""
        sections: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            safe_text = cls.ensure_utf8_text(chunk.text)
            safe_filename = cls.ensure_utf8_text(chunk.filename)
            sections.append(
                f"--- Context Excerpt {idx} ---\n"
                f"Filename: {safe_filename}\n"
                f"Page Number: {chunk.page_number}\n"
                f"Domain Category: {chunk.category}\n"
                f"Similarity Score: {chunk.similarity_score:.2%}\n"
                f"Content:\n{safe_text}\n"
            )
        return "\n".join(sections)

    @classmethod
    def _format_chat_history(cls, chat_history: list[dict[str, Any]] | None, max_turns: int = 4) -> str:
        """Include recent conversation turns to resolve follow-up pronouns cleanly."""
        if not chat_history:
            return "None"
        recent = chat_history[-(max_turns * 2) :]
        lines: list[str] = []
        for msg in recent:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = cls.ensure_utf8_text(str(msg.get("content", ""))).strip()
            if content:
                lines.append(f"{role}: {content[:500]}")
        return "\n".join(lines) if lines else "None"

    @staticmethod
    def _to_source_references(chunks: list[RetrievedChunk]) -> list[SourceReference]:
        """Convert RetrievedChunk objects into deduplicated display-ready SourceReference objects."""
        seen: set[tuple[str, int, int]] = set()
        refs: list[SourceReference] = []
        for chunk in chunks:
            key = (chunk.filename, chunk.page_number, chunk.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            refs.append(
                SourceReference(
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    category=chunk.category,
                    similarity_score=chunk.similarity_score,
                    snippet=chunk.text,
                )
            )
        return refs

    @classmethod
    def _verify_cited_sources(
        cls,
        answer_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> list[SourceReference]:
        """Keep only verified chunks corresponding to the context cited in the answer."""
        cited_pairs: set[tuple[str, int]] = set()
        pattern = re.compile(r"Source:\s*([^,\];]+?),\s*Pages?\s*(\d+)(?:\s*[-–]\s*(\d+))?")
        for match in pattern.finditer(answer_text):
            fname = match.group(1).strip()
            p_start = int(match.group(2))
            p_end = int(match.group(3)) if match.group(3) else p_start
            for page_num in range(min(p_start, p_end), max(p_start, p_end) + 1):
                cited_pairs.add((fname, page_num))

        if cited_pairs:
            matched_chunks = [
                c for c in retrieved_chunks if (c.filename, c.page_number) in cited_pairs
            ]
            if matched_chunks:
                return cls._to_source_references(matched_chunks)

        return cls._to_source_references(retrieved_chunks)

    def answer_question(
        self,
        question: str,
        category_filter: str | None = None,
        top_k: int | None = None,
        min_similarity: float | None = None,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline for a natural-language user question."""
        cleaned_question = self.ensure_utf8_text(question or "").strip()
        if not cleaned_question:
            return RAGResponse(
                answer="Please enter a question about your uploaded renewable energy documents.",
                found_in_context=False,
                model_used=self.model_name,
            )

        # Validate renewable-energy domain relevance before querying ChromaDB or Gemini
        is_relevant, retrieval_query = self.is_renewable_energy_question(
            cleaned_question, chat_history=chat_history
        )
        if not is_relevant:
            return RAGResponse(
                answer=OUT_OF_DOMAIN_MESSAGE,
                sources=[],
                found_in_context=False,
                domain_relevant=False,
                model_used=self.model_name,
                retrieval_count=0,
            )

        stats = self.vector_db.get_stats()
        if stats["total_chunks"] == 0:
            return RAGResponse(
                answer=(
                    "Your knowledge base is currently empty. Please upload renewable energy PDF or TXT "
                    "documents (or load the sample dataset) in the **Upload Documents** section first."
                ),
                found_in_context=False,
                model_used=self.model_name,
            )

        if not self.is_api_key_configured():
            return RAGResponse(
                answer=(
                    "Google Gemini API key is not configured. Please set `GEMINI_API_KEY` in your "
                    "`.env` file or enter your API key in the sidebar configuration."
                ),
                found_in_context=False,
                model_used=self.model_name,
                error="Missing GEMINI_API_KEY",
            )

        effective_top_k = top_k if top_k is not None else self.top_k
        effective_min_sim = min_similarity if min_similarity is not None else self.min_similarity

        try:
            query_embedding = self.embedding_manager.embed_query(retrieval_query)
            candidate_k = max(effective_top_k * 3, 12)
            candidate_chunks = self.vector_db.query_similar(
                query_embedding=query_embedding,
                top_k=candidate_k,
                category_filter=category_filter,
                min_similarity=effective_min_sim,
            )
            retrieved_chunks = self._rerank_and_filter_chunks(
                question=retrieval_query,
                chunks=candidate_chunks,
                top_k=effective_top_k,
            )
        except Exception as exc:
            return RAGResponse(
                answer=f"Failed to retrieve document context: {exc}",
                found_in_context=False,
                model_used=self.model_name,
                error=str(exc),
            )

        if not retrieved_chunks:
            filter_note = (
                f" (Category filter active: **{category_filter}**.)"
                if category_filter and category_filter != "All Categories"
                else ""
            )
            return RAGResponse(
                answer=f"{UNAVAILABLE_MESSAGE}{filter_note}",
                sources=[],
                found_in_context=False,
                domain_relevant=True,
                model_used=self.model_name,
                retrieval_count=0,
            )

        try:
            context_block = self._build_context_block(retrieved_chunks)
            history_block = self._format_chat_history(chat_history)
            safe_system_instruction = self.ensure_utf8_text(SYSTEM_INSTRUCTION)
            user_prompt = self.ensure_utf8_text(
                f"Recent Conversation History:\n{history_block}\n\n"
                f"Context Excerpts from Uploaded Documents:\n{context_block}\n\n"
                f"User Question: {cleaned_question}\n\n"
                "Instructions: Answer strictly using ONLY the Context Excerpts above, cite "
                "[Source: filename, Page X] for every claim, and if the excerpts do not contain "
                f"the answer, respond ONLY with:\n{UNAVAILABLE_MESSAGE}"
            )
            # Validate UTF-8 serialization explicitly (never encode prompts as ASCII)
            safe_system_instruction.encode("utf-8")
            user_prompt.encode("utf-8")
        except UnicodeError as exc:
            return RAGResponse(
                answer=f"Text encoding error preparing UTF-8 prompt for Gemini: {exc}",
                sources=[],
                found_in_context=False,
                model_used=self.model_name,
                retrieval_count=0,
                error=f"TextEncodingError: {exc}",
            )

        effective_api_key = self.get_effective_api_key()
        if not effective_api_key.isascii():
            return RAGResponse(
                answer=(
                    "Text encoding error: your `GEMINI_API_KEY` contains non-ASCII characters. "
                    "Please check that no smart quotes or dashes were pasted into `.env` or the sidebar."
                ),
                sources=[],
                found_in_context=False,
                model_used=self.model_name,
                retrieval_count=0,
                error="TextEncodingError: GEMINI_API_KEY contains non-ASCII characters",
            )

        candidate_models = [self.model_name] + [
            m for m in FALLBACK_GEMINI_MODELS if m != self.model_name
        ]
        last_exc: Exception | None = None
        encoding_exc: UnicodeError | None = None
        answer_text = ""
        used_model = self.model_name

        client = genai.Client(
            api_key=effective_api_key,
            http_options=types.HttpOptions(
                timeout=30000,
                headers={"Content-Type": "application/json; charset=utf-8"},
            ),
        )
        for candidate in candidate_models:
            try:
                response = client.models.generate_content(
                    model=candidate,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=safe_system_instruction,
                        temperature=0.1,
                    ),
                )
                answer_text = self.ensure_utf8_text(response.text or "").strip()
                used_model = candidate
                last_exc = None
                encoding_exc = None
                break
            except UnicodeError as exc:
                encoding_exc = exc
                last_exc = exc
                break
            except Exception as exc:
                last_exc = exc
                err_msg = str(exc)
                if any(
                    code in err_msg
                    for code in ("503", "504", "404", "UNAVAILABLE", "DEADLINE_EXCEEDED")
                ):
                    continue
                break

        if encoding_exc is not None:
            return RAGResponse(
                answer=(
                    f"Text encoding error sending UTF-8 request to Google Gemini API (`{self.model_name}`): "
                    f"{encoding_exc}"
                ),
                sources=[],
                found_in_context=False,
                model_used=self.model_name,
                retrieval_count=0,
                error=f"TextEncodingError: {encoding_exc}",
            )

        if last_exc is not None:
            api_error = GeminiAPIError(f"Google Gemini API error (`{self.model_name}`): {last_exc}")
            return RAGResponse(
                answer=(
                    f"Error calling Google Gemini API (`{self.model_name}`): {last_exc}\n\n"
                    "Please verify your `GEMINI_API_KEY`, network connectivity, and model access."
                ),
                sources=[],
                found_in_context=False,
                model_used=self.model_name,
                retrieval_count=0,
                error=str(api_error),
            )

        if not answer_text:
            return RAGResponse(
                answer=UNAVAILABLE_MESSAGE,
                sources=[],
                found_in_context=False,
                model_used=used_model,
                retrieval_count=0,
            )

        # If Gemini indicates the context lacks the answer, suppress unrelated sources
        lowered_ans = answer_text.lower()
        if (
            "do not contain relevant information" in lowered_ans
            or "does not contain relevant information" in lowered_ans
        ):
            return RAGResponse(
                answer=UNAVAILABLE_MESSAGE,
                sources=[],
                found_in_context=False,
                domain_relevant=True,
                model_used=used_model,
                retrieval_count=0,
            )

        verified_sources = self._verify_cited_sources(answer_text, retrieved_chunks)

        return RAGResponse(
            answer=answer_text,
            sources=verified_sources,
            found_in_context=True,
            domain_relevant=True,
            model_used=used_model,
            retrieval_count=len(verified_sources),
        )
