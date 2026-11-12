"""Document ingestion, text extraction, cleaning, chunking, and domain classification."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


class PDFProcessingError(ValueError):
    """Raised when reading or extracting text from a PDF document fails."""


class TextEncodingError(ValueError):
    """Raised when decoding or normalizing document text into valid UTF-8 fails."""


RENEWABLE_CATEGORIES: list[str] = [
    "Solar Energy",
    "Wind Energy",
    "Hydroelectric Energy",
    "Geothermal Energy",
    "Biomass Energy",
    "Green Hydrogen",
    "Renewable Energy Storage",
    "Government Renewable Energy Schemes",
]

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Solar Energy": (
        "solar",
        "photovoltaic",
        "pv",
        "monocrystalline",
        "polycrystalline",
        "perovskite",
        "inverter",
        "mppt",
        "irradiance",
        "csp",
        "rooftop",
    ),
    "Wind Energy": (
        "wind",
        "turbine",
        "offshore",
        "onshore",
        "rotor",
        "nacelle",
        "betz",
        "aerodynamic",
        "wind farm",
        "hub height",
    ),
    "Hydroelectric Energy": (
        "hydro",
        "hydroelectric",
        "hydropower",
        "dam",
        "reservoir",
        "run-of-river",
        "francis",
        "pelton",
        "kaplan",
        "penstock",
    ),
    "Geothermal Energy": (
        "geothermal",
        "magma",
        "binary cycle",
        "flash steam",
        "dry steam",
        "egs",
        "enhanced geothermal",
        "subsurface heat",
        "ground source",
    ),
    "Biomass Energy": (
        "biomass",
        "biofuel",
        "biogas",
        "anaerobic digestion",
        "bioethanol",
        "biodiesel",
        "lignocellulosic",
        "gasification",
        "pyrolysis",
        "feedstock",
    ),
    "Green Hydrogen": (
        "hydrogen",
        "electrolyzer",
        "electrolysis",
        "pem",
        "alkaline",
        "soec",
        "fuel cell",
        "green ammonia",
        "h2",
        "decarbonization",
    ),
    "Renewable Energy Storage": (
        "storage",
        "battery",
        "bess",
        "lithium",
        "lfp",
        "flow battery",
        "vanadium",
        "pumped hydro",
        "compressed air",
        "round-trip efficiency",
        "sodium-ion",
    ),
    "Government Renewable Energy Schemes": (
        "scheme",
        "policy",
        "policies",
        "subsidy",
        "kusum",
        "surya ghar",
        "pli",
        "mission",
        "incentive",
        "tariff",
        "ministry",
        "mnre",
        "ira",
        "green deal",
        "net metering",
        "mandate",
        "mandates",
        "obligation",
        "corridor",
        "cbam",
        "almm",
        "rpo",
        "rgo",
        "ists",
        "tax credit",
        "regulatory",
    ),
}


@dataclass
class DocumentChunk:
    """Represents a single cleaned and chunked text segment with full source metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, str | int] = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """Result returned after extracting and chunking an uploaded or local document."""

    filename: str
    file_hash: str
    file_type: str
    page_count: int
    category: str
    chunks: list[DocumentChunk]
    uploaded_at: str


class DocumentProcessor:
    """Extracts, cleans, chunks, and categorizes PDF and TXT documents."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def compute_file_hash(raw_bytes: bytes) -> str:
        """Compute deterministic SHA-256 hash of file contents."""
        return hashlib.sha256(raw_bytes).hexdigest()

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Normalize an uploaded filename to prevent path traversal and invalid chars."""
        base = Path(filename).name
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")
        return sanitized or f"document_{int(datetime.now(timezone.utc).timestamp())}.txt"

    @staticmethod
    def ensure_utf8_text(text: str) -> str:
        """Ensure string is valid UTF-8 Unicode while preserving symbols such as – — ‘ ’ \" ° × ≥ ≤."""
        if not text:
            return ""
        try:
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize Unicode safely to UTF-8, fix line-break hyphenation, and preserve special symbols."""
        if not text:
            return ""
        safe_text = DocumentProcessor.ensure_utf8_text(text)
        normalized = unicodedata.normalize("NFKC", safe_text)
        # Remove control characters except newline and tab
        normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", normalized)
        # Join words broken across PDF line wraps: "photo-\nvoltaic" -> "photovoltaic"
        normalized = re.sub(r"([A-Za-z])-\s*\n\s*([A-Za-z])", r"\1\2", normalized)
        # Normalize line endings and collapse excessive blank lines
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        # Trim whitespace on individual lines and drop blank lines
        lines = [line.strip() for line in normalized.split("\n")]
        cleaned = "\n".join(line for line in lines if line)
        # Rejoin mid-sentence PDF line wraps while keeping headings/sentences on separate lines
        cleaned = re.sub(r"(?<![.!?:;\n])\n(?=[a-z0-9(])", " ", cleaned)
        cleaned = re.sub(r"(?<=[,;])\n(?=[A-Za-z])", " ", cleaned)
        return DocumentProcessor.ensure_utf8_text(cleaned.strip())

    @staticmethod
    def detect_category(text: str, filename: str = "") -> str:
        """Classify text into one of the 8 renewable energy categories using word-boundary scoring."""
        text_haystack = (text[:8000] or "").lower()
        scores: dict[str, int] = {}
        for category, keywords in _CATEGORY_KEYWORDS.items():
            score = sum(
                len(re.findall(rf"\b{re.escape(kw)}\b", text_haystack))
                for kw in keywords
            )
            scores[category] = score

        best_score = max(scores.values())
        if best_score > 0:
            # Break ties in text score using filename keywords if available
            tied = [cat for cat, sc in scores.items() if sc == best_score]
            if len(tied) == 1 or not filename:
                return tied[0]
            name_haystack = filename.lower()
            return max(
                tied,
                key=lambda cat: sum(
                    len(re.findall(rf"\b{re.escape(kw)}\b", name_haystack))
                    for kw in _CATEGORY_KEYWORDS[cat]
                ),
            )

        if filename:
            name_haystack = filename.lower()
            name_scores = {
                category: sum(
                    len(re.findall(rf"\b{re.escape(kw)}\b", name_haystack))
                    for kw in keywords
                )
                for category, keywords in _CATEGORY_KEYWORDS.items()
            }
            best_name_cat = max(name_scores, key=lambda cat: name_scores[cat])
            if name_scores[best_name_cat] > 0:
                return best_name_cat

        return "General Renewable Energy"

    def split_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks along sentence and paragraph boundaries."""
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                # Prioritize paragraph & sentence breaks before falling back to line/word breaks
                window = text[start:end]
                min_break = max(int(self.chunk_size * 0.45), 1)
                split_offset = -1
                for sep in ("\n\n", ".\n", ". ", "?\n", "? ", "!\n", "! ", "\n", "; ", " "):
                    idx = window.rfind(sep, min_break)
                    if idx != -1:
                        split_offset = idx + len(sep)
                        break
                if split_offset > 0:
                    end = start + split_offset

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= text_len:
                break

            target_start = max(end - self.chunk_overlap, start + 1)
            next_start = target_start

            # Prefer starting the overlapping chunk on a complete sentence boundary
            search_lo = max(target_start - 50, start + 1)
            search_hi = min(end - 20, text_len)
            sentence_aligned = False
            if search_hi > search_lo:
                overlap_slice = text[search_lo:search_hi]
                for sep in (".\n", ". ", "?\n", "? ", "!\n", "! ", "\n"):
                    rel_idx = overlap_slice.rfind(sep)
                    if rel_idx != -1:
                        candidate = search_lo + rel_idx + len(sep)
                        if start < candidate < end:
                            next_start = candidate
                            sentence_aligned = True
                            break

            # Fallback: snap forward to a whitespace boundary so words are never sliced
            if (
                not sentence_aligned
                and next_start > 0
                and next_start < text_len
                and not text[next_start - 1].isspace()
            ):
                space_idx = text.find(" ", next_start, min(next_start + 35, text_len))
                if space_idx != -1:
                    next_start = space_idx + 1

            start = max(next_start, start + 1)

        return chunks

    def _extract_pdf_pages(self, file_path: Path) -> tuple[list[tuple[int, str]], int]:
        """Extract cleaned UTF-8 text page-by-page from a PDF file and return (pages, total_pages)."""
        pages: list[tuple[int, str]] = []
        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            raise PDFProcessingError(f"PDF processing error opening '{file_path.name}': {exc}") from exc

        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
                if decrypt_result == 0:
                    raise PDFProcessingError(
                        f"PDF processing error: '{file_path.name}' is password-protected."
                    )
            except PDFProcessingError:
                raise
            except Exception as exc:
                raise PDFProcessingError(
                    f"PDF processing error: '{file_path.name}' is password-protected ({exc})."
                ) from exc

        total_pages = len(reader.pages)
        page_errors: list[str] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                raw_text = page.extract_text() or ""
            except Exception as exc:
                page_errors.append(f"page {page_idx}: {exc}")
                continue
            try:
                cleaned = self.clean_text(raw_text)
            except UnicodeError as exc:
                raise TextEncodingError(
                    f"Text encoding error normalizing page {page_idx} of '{file_path.name}': {exc}"
                ) from exc
            if cleaned:
                pages.append((page_idx, cleaned))

        if not pages and page_errors:
            raise PDFProcessingError(
                f"PDF processing error extracting '{file_path.name}': {'; '.join(page_errors[:3])}"
            )

        return pages, total_pages

    def _extract_txt_pages(self, raw_bytes: bytes, filename: str) -> list[tuple[int, str]]:
        """Decode and clean a TXT document using UTF-8 first with clean fallbacks."""
        decoded = ""
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
            try:
                decoded = raw_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if not decoded:
            raise TextEncodingError(
                f"Text encoding error: unable to decode '{filename}' as UTF-8 or supported encodings."
            )

        try:
            cleaned = self.clean_text(decoded)
        except UnicodeError as exc:
            raise TextEncodingError(
                f"Text encoding error normalizing '{filename}': {exc}"
            ) from exc
        return [(1, cleaned)] if cleaned else []

    def save_uploaded_bytes(self, filename: str, raw_bytes: bytes, target_dir: Path) -> Path:
        """Persist uploaded file bytes safely under target_dir and return the saved Path."""
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.sanitize_filename(filename)
        dest_path = target_dir / safe_name
        dest_path.write_bytes(raw_bytes)
        return dest_path

    def process_file(
        self,
        file_path: Path,
        category: str = "Auto-Detect",
        raw_bytes: bytes | None = None,
    ) -> ProcessedDocument:
        """Extract, clean, chunk, and build metadata for a PDF or TXT file on disk."""
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = file_path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{ext}' for '{file_path.name}'. Allowed: PDF, TXT."
            )

        content_bytes = raw_bytes if raw_bytes is not None else file_path.read_bytes()
        if not content_bytes:
            raise ValueError(f"Uploaded file '{file_path.name}' is empty.")

        file_hash = self.compute_file_hash(content_bytes)
        if ext == ".pdf":
            pages, total_pages = self._extract_pdf_pages(file_path)
        else:
            pages = self._extract_txt_pages(content_bytes, file_path.name)
            total_pages = 1

        if not pages:
            err_cls = PDFProcessingError if ext == ".pdf" else ValueError
            raise err_cls(
                f"No extractable text found in '{file_path.name}'. "
                "If this is a scanned image PDF, please run OCR first."
            )

        auto_detect = not category or category == "Auto-Detect"
        uploaded_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        chunks: list[DocumentChunk] = []
        detected_categories: list[str] = []
        chunk_counter = 0

        for page_num, page_text in pages:
            page_category = (
                self.detect_category(page_text, file_path.name) if auto_detect else category
            )
            if page_category not in detected_categories:
                detected_categories.append(page_category)

            page_chunks = self.split_text(page_text)
            for segment in page_chunks:
                chunk_id = f"{file_hash[:12]}_p{page_num}_c{chunk_counter}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=segment,
                        metadata={
                            "filename": file_path.name,
                            "file_hash": file_hash,
                            "file_type": ext.lstrip("."),
                            "page_number": page_num,
                            "total_pages": total_pages,
                            "chunk_index": chunk_counter,
                            "category": page_category,
                            "uploaded_at": uploaded_at,
                        },
                    )
                )
                chunk_counter += 1

        resolved_category = ", ".join(detected_categories) if detected_categories else "General Renewable Energy"

        return ProcessedDocument(
            filename=file_path.name,
            file_hash=file_hash,
            file_type=ext.lstrip("."),
            page_count=total_pages,
            category=resolved_category,
            chunks=chunks,
            uploaded_at=uploaded_at,
        )

    @staticmethod
    def list_disk_documents(documents_dir: Path) -> list[Path]:
        """List supported PDF and TXT documents stored in documents_dir."""
        if not documents_dir.exists():
            return []
        files: Iterable[Path] = sorted(documents_dir.iterdir(), key=lambda p: p.name.lower())
        return [
            f
            for f in files
            if f.is_file() and f.suffix.lower() in DocumentProcessor.SUPPORTED_EXTENSIONS
        ]
