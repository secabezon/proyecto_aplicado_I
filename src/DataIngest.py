from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger
from PyPDF2 import PdfReader

from LLM_prompt import llm_prompt, safe_json_loads
from project_config import cfg, project_root


def clean_text(text: str) -> str:
    """
    Normalize extracted text by collapsing whitespace.

    Args:
        text: Raw text.

    Returns:
        Cleaned text.
    """
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_words(words: List[str], words_per_chunk: int, overlap: int) -> List[str]:
    """
    Chunk a list of words using a fixed window size and overlap.

    Args:
        words: Tokenized words.
        words_per_chunk: Chunk size in words.
        overlap: Overlap size in words.

    Returns:
        A list of chunk strings.

    Raises:
        ValueError: If parameters are invalid.
    """
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be > 0")
    if overlap < 0 or overlap >= words_per_chunk:
        raise ValueError("overlap must be >= 0 and < words_per_chunk")

    chunks: List[str] = []
    step = words_per_chunk - overlap
    for start in range(0, len(words), step):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(words):
            break
    return chunks


def pdf_pages_to_text(pdf_path: Path, max_pages: Optional[int] = None) -> str:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF.
        max_pages: Optional max number of pages to extract from the start.

    Returns:
        Extracted text.
    """
    reader = PdfReader(str(pdf_path))
    n_pages = len(reader.pages)
    limit = min(n_pages, max_pages) if max_pages else n_pages
    parts: List[str] = []

    for i in range(limit):
        parts.append(reader.pages[i].extract_text() or "")

    return "\n".join(parts)


def heuristic_year(text: str) -> Optional[str]:
    """
    Detect a plausible publication year in text.

    Args:
        text: Text to scan.

    Returns:
        A 4-digit year if found, otherwise None.
    """
    matches = re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")
    return matches[0] if matches else None


def extract_paper_metadata(pdf_path: Path) -> Tuple[str, str]:
    """
    Extract (paper_date, paper_title) using an LLM with a heuristic fallback.

    Args:
        pdf_path: PDF path.

    Returns:
        (paper_date, paper_title)
    """
    pages = int(getattr(cfg, "PAPER_META_PAGES", 2))
    max_chars = int(getattr(cfg, "PAPER_META_MAX_CHARS", 9000))

    raw_head = pdf_pages_to_text(pdf_path, max_pages=pages)
    head = clean_text(raw_head)[:max_chars]

    prompt = getattr(cfg, "PROMPT_PAPER_METADATA").format(text=head)
    messages = [{"role": "user", "content": prompt}]

    paper_date = "UNKNOWN"
    paper_title = "UNKNOWN"

    try:
        out = llm_prompt(messages, temperature=float(getattr(cfg, "LLM_TEMPERATURE", 0.0)))
        data = safe_json_loads(out)
        paper_title_val = data.get("paper_title")
        paper_date_val = data.get("paper_date")

        if isinstance(paper_title_val, str) and paper_title_val.strip():
            paper_title = paper_title_val.strip()

        if isinstance(paper_date_val, str) and paper_date_val.strip():
            paper_date = paper_date_val.strip()

    except Exception:
        pass

    if paper_date == "UNKNOWN":
        guess_year = heuristic_year(head)
        if guess_year:
            paper_date = guess_year

    if paper_title == "UNKNOWN":
        paper_title = pdf_path.stem.replace("_", " ").strip() or "UNKNOWN"

    return paper_date, paper_title


def build_output_paths() -> Tuple[Path, Path]:
    """
    Build deterministic output paths.

    Returns:
        (chunks_json_path, metadata_json_path)
    """
    root = project_root()
    out_dir = root / getattr(cfg, "OUTPUT_DIR", "outputs/ingest")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = getattr(cfg, "BASE_OUTPUT_NAME", "corpus_chunks")
    chunks_path = out_dir / f"{base}.json"
    meta_path = out_dir / f"{base}_metadata.json"
    return chunks_path, meta_path


def main() -> Dict[str, str]:
    """
    Ingest all PDFs from cfg.INPUT_DIR and write a single JSON list of chunk records.

    Each chunk record includes:
        doc_id, chunk_id, order, text, paper_date, paper_title

    Returns:
        Paths of generated files.
    """
    root = project_root()
    input_dir = root / getattr(cfg, "INPUT_DIR", "data/raw")
    words_per_chunk = int(getattr(cfg, "WORDS_PER_CHUNK", 120))
    overlap = int(getattr(cfg, "OVERLAP", 40))

    date_key = getattr(cfg, "PAPER_DATE_KEY", "paper_date")
    title_key = getattr(cfg, "PAPER_TITLE_KEY", "paper_title")

    chunks_file, metadata_file = build_output_paths()

    pdf_paths = sorted(input_dir.rglob("*.pdf"))
    logger.info(
        "Starting PDF to chunks conversion | input_dir='{}' | output_file='{}' | n_pdfs={}",
        str(input_dir),
        str(chunks_file),
        len(pdf_paths),
    )

    all_records: List[Dict[str, object]] = []
    per_doc_stats: List[Dict[str, object]] = []

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        logger.info("Processing file {}/{}: {}", idx, len(pdf_paths), pdf_path.name)

        doc_id = pdf_path.stem
        paper_date, paper_title = extract_paper_metadata(pdf_path)

        raw_text = pdf_pages_to_text(pdf_path, max_pages=None)
        cleaned = clean_text(raw_text)

        words = cleaned.split()
        chunks = chunk_words(words, words_per_chunk=words_per_chunk, overlap=overlap)

        for order, chunk_text in enumerate(chunks):
            all_records.append(
                {
                    "chunk_id": f"{doc_id}_chunk_{order:04d}",
                    "doc_id": doc_id,
                    "order": order,
                    "text": chunk_text,
                    date_key: paper_date,
                    title_key: paper_title,
                }
            )

        per_doc_stats.append(
            {
                "doc_id": doc_id,
                "pdf_file": str(pdf_path),
                "num_chars_clean": len(cleaned),
                "num_chunks": len(chunks),
                date_key: paper_date,
                title_key: paper_title,
            }
        )

    chunks_file.write_text(
        json.dumps(all_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "num_pdfs": len(pdf_paths),
        "words_per_chunk": words_per_chunk,
        "overlap": overlap,
        "num_total_chunks": len(all_records),
        "per_doc_stats": per_doc_stats,
    }
    metadata_file.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.success(
        "Finished writing {} chunk records to '{}' and metadata to '{}'.",
        len(all_records),
        str(chunks_file),
        str(metadata_file),
    )

    return {"chunks_file": str(chunks_file), "metadata_file": str(metadata_file)}
