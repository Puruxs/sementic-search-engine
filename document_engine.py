"""
document_engine.py - 100% Private, Local Document Search Engine & Book Intelligence.
Advanced features:
1. Book PDF Processing: Chapter/TOC hierarchy, multi-page indexing, streaming extraction.
2. Structured Tabular Retrieval: Native PyMuPDF table finder (`page.find_tables()`),
   Word table extraction, and direct Excel (.xlsx) / CSV tabular data ingestion.
3. Tabular Query Intent Boosting for financial & numerical precision.
4. Intelligent Stratified Summarization for long books and large reports.
"""

import hashlib
import io
import math
import os
import re
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageEnhance

# PDF libraries
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

try:
    import pypdf
except ImportError:
    pypdf = None

# DOCX library
try:
    import docx
except ImportError:
    docx = None

# Tabular Data Libraries
try:
    import pandas as pd
except ImportError:
    pd = None

# OCR library
try:
    import pytesseract
except ImportError:
    pytesseract = None


class DocumentEngine:
    """High-performance in-memory document engine supporting books, PDFs, tables, and docs."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 140):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents: Dict[str, Dict] = {}  # filename -> metadata
        self.chunks: List[Dict] = []  # list of chunk dicts
        self.file_hashes: Dict[str, str] = {}  # filename -> sha256 hash
        self.book_tocs: Dict[str, List] = {}  # filename -> Table of Contents
        self.total_tokens: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}  # term -> document/chunk frequency

    def clear(self):
        """Clears all stored documents, chunks, and indexes."""
        self.documents.clear()
        self.chunks.clear()
        self.file_hashes.clear()
        self.book_tocs.clear()
        self.total_tokens = 0
        self.avg_doc_len = 0.0
        self.doc_freqs.clear()

    def remove_document(self, filename: str) -> bool:
        """Removes a single document and its chunks from the index."""
        if filename not in self.documents:
            return False
        del self.documents[filename]
        if filename in self.file_hashes:
            del self.file_hashes[filename]
        if filename in self.book_tocs:
            del self.book_tocs[filename]

        self.chunks = [c for c in self.chunks if c["filename"] != filename]
        for idx, c in enumerate(self.chunks, start=1):
            c["chunk_id"] = idx
        self._recompute_stats()
        return True

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalizes whitespace, line breaks, and non-printable characters."""
        if not text:
            return ""
        text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Fast alphanumeric tokenizer for Okapi BM25."""
        return re.findall(r"\b[a-zA-Z0-9_\-\.\%]{2,}\b", text.lower())

    def extract_from_pdf(self, file_bytes: bytes, filename: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Extracts both text and structured tables from PDF books/documents.
        Also captures the Table of Contents (TOC) hierarchy if present.
        """
        pages_content = []
        book_toc = []

        if fitz:
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")

                # Extract Table of Contents / Book Outline
                try:
                    raw_toc = doc.get_toc()  # [[level, title, page_number], ...]
                    if raw_toc:
                        book_toc = [{"level": item[0], "title": item[1].strip(), "page": item[2]} for item in raw_toc]
                except Exception:
                    pass

                # Track active chapter based on TOC
                current_chapter = "General"

                for page_idx, page in enumerate(doc, start=1):
                    # Check if this page matches a chapter start
                    for item in book_toc:
                        if item["page"] == page_idx:
                            current_chapter = item["title"]

                    page_elements = []

                    # 1. Extract Structured Tables via PyMuPDF table finder
                    table_markdowns = []
                    try:
                        tabs = page.find_tables()
                        if tabs and tabs.tables:
                            for tab_idx, tab in enumerate(tabs.tables, start=1):
                                md = tab.to_markdown()
                                if md and len(md.strip()) > 20:
                                    table_markdowns.append(
                                        f"\n[📊 TABLE {tab_idx} ON PAGE {page_idx}]:\n{md.strip()}\n"
                                    )
                    except Exception:
                        pass

                    # 2. Extract Body Text
                    raw_text = page.get_text() or ""
                    clean_body = self._clean_text(raw_text)

                    # If page text is empty (scanned PDF / handwritten / photocopied notes), attempt OCR
                    if not clean_body:
                        if pytesseract:
                            try:
                                pix = page.get_pixmap(dpi=150)
                                img = Image.open(io.BytesIO(pix.tobytes("png")))
                                ocr_res = pytesseract.image_to_string(img)
                                clean_body = self._clean_text(ocr_res)
                            except Exception:
                                pass

                    # If still empty but page has images, record page description
                    if not clean_body:
                        img_count = len(page.get_images())
                        if img_count > 0:
                            clean_body = f"[Page {page_idx} of {filename}: Scanned image/diagram content ({img_count} visual element(s))]"

                    # Combine body and structured tables
                    combined = clean_body
                    if table_markdowns:
                        combined += "\n\n" + "\n".join(table_markdowns)

                    if combined.strip():
                        pages_content.append({
                            "page": page_idx,
                            "chapter": current_chapter,
                            "text": combined.strip(),
                            "has_table": len(table_markdowns) > 0,
                        })

                if pages_content:
                    return pages_content, book_toc
            except Exception:
                pass

        # Fallback to PyPDF
        if pypdf:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                for page_idx, page in enumerate(reader.pages, start=1):
                    raw_text = page.extract_text() or ""
                    clean_body = self._clean_text(raw_text)
                    if clean_body:
                        pages_content.append({
                            "page": page_idx,
                            "chapter": f"Page {page_idx}",
                            "text": clean_body,
                            "has_table": False,
                        })
                return pages_content, []
            except Exception:
                pass

        return pages_content, book_toc

    def extract_from_docx(self, file_bytes: bytes, filename: str) -> List[Dict]:
        """Extracts text and structured tables from DOCX documents."""
        if not docx:
            return []

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            sections = []
            has_table = False

            # Extract body paragraphs
            for p in doc.paragraphs:
                cleaned = p.text.strip()
                if cleaned:
                    sections.append(cleaned)

            # Extract structured tables as markdown tables
            for t_idx, table in enumerate(doc.tables, start=1):
                table_rows = []
                for row in table.rows:
                    row_cells = [self._clean_text(c.text) for c in row.cells]
                    if any(row_cells):
                        table_rows.append("| " + " | ".join(row_cells) + " |")

                if len(table_rows) >= 2:
                    has_table = True
                    # Insert markdown separator after header row
                    num_cols = len(table.columns)
                    separator = "| " + " | ".join(["---"] * num_cols) + " |"
                    table_rows.insert(1, separator)
                    sections.append(f"\n[📊 TABLE {t_idx}]:\n" + "\n".join(table_rows) + "\n")

            full_text = self._clean_text("\n\n".join(sections))
            if full_text:
                return [{
                    "page": 1,
                    "chapter": "Document Body",
                    "text": full_text,
                    "has_table": has_table,
                }]
        except Exception:
            pass

        return []

    def extract_from_tabular(self, file_bytes: bytes, filename: str) -> List[Dict]:
        """Extracts structured tabular data from Excel (.xlsx/.xls) and CSV (.csv) files."""
        if not pd:
            return []

        ext = os.path.splitext(filename)[1].lower()
        pages = []

        try:
            if ext == ".csv":
                df = pd.read_csv(io.BytesIO(file_bytes))
                num_rows, num_cols = df.shape
                # Summary header
                summary_header = f"### [📊 TABULAR DATASET: {filename}] ({num_rows} rows x {num_cols} columns)\n"
                summary_header += f"Columns: {', '.join([str(c) for c in df.columns])}\n\n"

                # Convert dataframe to markdown representation (in blocks of 30 rows if large)
                chunk_step = 30
                for start_row in range(0, min(num_rows, 300), chunk_step):
                    end_row = min(start_row + chunk_step, num_rows)
                    sub_df = df.iloc[start_row:end_row]
                    md_table = sub_df.to_markdown(index=False)
                    page_text = (
                        f"{summary_header}Rows {start_row + 1} to {end_row} of {num_rows}:\n{md_table}"
                    )
                    pages.append({
                        "page": (start_row // chunk_step) + 1,
                        "chapter": f"Rows {start_row + 1}-{end_row}",
                        "text": page_text,
                        "has_table": True,
                    })

            elif ext in (".xlsx", ".xls"):
                excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
                for sheet_idx, sheet_name in enumerate(excel_file.sheet_names, start=1):
                    df = excel_file.parse(sheet_name)
                    num_rows, num_cols = df.shape
                    sheet_summary = f"### [📊 SHEET: '{sheet_name}' in {filename}] ({num_rows} rows x {num_cols} columns)\n"
                    sheet_summary += f"Headers: {', '.join([str(c) for c in df.columns])}\n\n"

                    chunk_step = 30
                    for start_row in range(0, min(num_rows, 300), chunk_step):
                        end_row = min(start_row + chunk_step, num_rows)
                        sub_df = df.iloc[start_row:end_row]
                        md_table = sub_df.to_markdown(index=False)
                        pages.append({
                            "page": sheet_idx,
                            "chapter": f"Sheet: {sheet_name}",
                            "text": f"{sheet_summary}Rows {start_row + 1}-{end_row}:\n{md_table}",
                            "has_table": True,
                        })
        except Exception:
            pass

        return pages

    def extract_from_image(self, file_bytes: bytes, filename: str) -> List[Dict]:
        """Extracts text from images using preprocessed OCR (pytesseract) or descriptive metadata."""
        extracted_text = ""

        try:
            raw_img = Image.open(io.BytesIO(file_bytes))
            width, height = raw_img.size

            if pytesseract:
                try:
                    gray_img = raw_img.convert("L")
                    enhancer = ImageEnhance.Contrast(gray_img)
                    enhanced_img = enhancer.enhance(1.8)
                    ocr_res = pytesseract.image_to_string(enhanced_img)
                    if ocr_res and len(ocr_res.strip()) > 10:
                        extracted_text = self._clean_text(ocr_res)
                except Exception:
                    pass

            if not extracted_text:
                extracted_text = (
                    f"[Visual Image File: {filename} • Format: {raw_img.format} • "
                    f"Dimensions: {width}x{height}px]. Uploaded as image context for analysis."
                )

            return [{
                "page": 1,
                "chapter": "Image Content",
                "text": extracted_text,
                "has_table": False,
            }]
        except Exception:
            return [{"page": 1, "chapter": "Image", "text": f"[Uploaded Image: {filename}]", "has_table": False}]

    def extract_from_txt(self, file_bytes: bytes, filename: str) -> List[Dict]:
        """Extracts text from raw text / markdown / code files."""
        for enc in ("utf-8", "utf-16", "latin-1", "cp1252"):
            try:
                raw = file_bytes.decode(enc)
                cleaned = self._clean_text(raw)
                if cleaned:
                    return [{"page": 1, "chapter": "Text Document", "text": cleaned, "has_table": False}]
            except (UnicodeDecodeError, Exception):
                continue
        return []

    def load_document(self, file_bytes: bytes, filename: str) -> Tuple[bool, str, int]:
        """
        Parses and indexes books, PDFs, spreadsheets, Word docs, and images.
        Deduplicates by SHA-256 to avoid redundant parsing.
        """
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if filename in self.documents and self.file_hashes.get(filename) == file_hash:
            return True, f"'{filename}' is already indexed.", self.documents[filename]["chunks_count"]

        if filename in self.documents:
            self.remove_document(filename)

        ext = os.path.splitext(filename)[1].lower()
        pages: List[Dict] = []
        book_toc: List[Dict] = []

        if ext == ".pdf":
            pages, book_toc = self.extract_from_pdf(file_bytes, filename)
        elif ext in (".docx", ".doc"):
            pages = self.extract_from_docx(file_bytes, filename)
        elif ext in (".xlsx", ".xls", ".csv"):
            pages = self.extract_from_tabular(file_bytes, filename)
        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            pages = self.extract_from_image(file_bytes, filename)
        elif ext in (".txt", ".md", ".json", ".py", ".html", ".xml"):
            pages = self.extract_from_txt(file_bytes, filename)
        else:
            return False, f"Unsupported file format: '{ext}'", 0

        if not pages:
            return False, f"No readable text or tabular data found in '{filename}'.", 0

        # Intelligent sliding-window chunking
        doc_chunks = []
        total_tables = 0

        for page_info in pages:
            page_num = page_info["page"]
            chapter = page_info.get("chapter", f"Page {page_num}")
            text = page_info["text"]
            has_table = page_info.get("has_table", False)
            if has_table:
                total_tables += 1

            start = 0
            text_len = len(text)

            while start < text_len:
                end = min(start + self.chunk_size, text_len)

                # Keep markdown tables intact when possible
                if end < text_len:
                    sub = text[start:end]
                    match = re.search(r"(\n\n|\.\s+|\?\s+|\!\s+)[^\n\.\?\!]*$", sub)
                    if match and match.start() > (self.chunk_size // 3):
                        end = start + match.start() + 1
                    else:
                        last_space = text.rfind(" ", start, end)
                        if last_space > start + (self.chunk_size // 2):
                            end = last_space

                chunk_text = text[start:end].strip()
                if len(chunk_text) >= 25:
                    tokens = self._tokenize(chunk_text)
                    is_table_chunk = ("|" in chunk_text and "---" in chunk_text) or "[📊" in chunk_text
                    doc_chunks.append({
                        "chunk_id": len(self.chunks) + len(doc_chunks) + 1,
                        "filename": filename,
                        "file_type": ext.lstrip(".").upper(),
                        "page": page_num,
                        "chapter": chapter,
                        "text": chunk_text,
                        "tokens": tokens,
                        "token_count": len(tokens),
                        "is_table": is_table_chunk,
                    })

                if end >= text_len:
                    break
                start = max(start + 1, end - self.chunk_overlap)

        if not doc_chunks:
            return False, f"Content in '{filename}' was too short for indexing.", 0

        self.documents[filename] = {
            "filename": filename,
            "file_type": ext.lstrip(".").upper(),
            "pages": len(pages),
            "chunks_count": len(doc_chunks),
            "tables_count": total_tables,
            "size_bytes": len(file_bytes),
            "has_toc": len(book_toc) > 0,
        }
        self.file_hashes[filename] = file_hash
        if book_toc:
            self.book_tocs[filename] = book_toc

        self.chunks.extend(doc_chunks)
        self._recompute_stats()

        msg = f"Indexed '{filename}' ({len(pages)} pages/sheets, {len(doc_chunks)} chunks"
        if total_tables > 0:
            msg += f", {total_tables} tables extracted"
        if book_toc:
            msg += f", {len(book_toc)} chapters in TOC"
        msg += ")."

        return True, msg, len(doc_chunks)

    def _recompute_stats(self):
        """Recomputes BM25 collection statistics."""
        self.total_tokens = sum(c["token_count"] for c in self.chunks)
        self.avg_doc_len = (self.total_tokens / len(self.chunks)) if self.chunks else 0.0

        self.doc_freqs.clear()
        for c in self.chunks:
            for term in set(c["tokens"]):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieves top-K most relevant chunks using Okapi BM25 ranking.
        Includes Tabular Intent Boosting for numerical and table queries.
        """
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Detect tabular intent
        tabular_keywords = {
            "table", "data", "revenue", "price", "metric", "cost", "salary",
            "number", "figure", "growth", "total", "rate", "sum", "percent",
            "percentage", "%", "quarter", "annual", "sheet", "column", "row", "stats",
        }
        has_tabular_intent = any(w in tabular_keywords for w in query_tokens) or any(char.isdigit() for char in query)

        num_chunks = len(self.chunks)
        k1 = 1.5
        b = 0.75

        scores = []
        for c in self.chunks:
            score = 0.0
            doc_len = c["token_count"]
            chunk_tokens = c["tokens"]
            term_counts = {}
            for t in chunk_tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for q_term in query_tokens:
                if q_term in term_counts:
                    tf = term_counts[q_term]
                    df = self.doc_freqs.get(q_term, 0)
                    idf = math.log((num_chunks - df + 0.5) / (df + 0.5) + 1.0)
                    numerator = tf * (k1 + 1.0)
                    denominator = tf + k1 * (1.0 - b + b * (doc_len / (self.avg_doc_len or 1.0)))
                    score += idf * (numerator / denominator)

            # Boost tabular chunks if query asks about tables, numbers, or metrics
            if has_tabular_intent and c.get("is_table"):
                score *= 1.4

            scores.append(score)

        # Rank pairs
        ranked_pairs = sorted(zip(scores, self.chunks), key=lambda x: x[0], reverse=True)

        results = []
        result_id = 1
        for raw_score, chunk in ranked_pairs:
            if raw_score <= 0.0 and len(results) >= 2:
                break

            norm_pct = int(min(99, max(45, (raw_score / (raw_score + 4.5)) * 100))) if raw_score > 0 else 50
            chapter_info = f" • {chunk.get('chapter')}" if chunk.get("chapter") and chunk.get("chapter") != "General" else ""
            title = f"{chunk['filename']} (Page {chunk['page']}{chapter_info})"

            results.append({
                "id": result_id,
                "filename": chunk["filename"],
                "file_type": chunk["file_type"],
                "page": chunk["page"],
                "chapter": chunk.get("chapter", ""),
                "title": title,
                "domain": f"local::{chunk['file_type'].lower()}",
                "url": f"local://{chunk['filename']}#page={chunk['page']}",
                "snippet": chunk["text"],
                "content": chunk["text"],
                "score": raw_score,
                "match_pct": f"{norm_pct}%",
                "is_table": chunk.get("is_table", False),
                "is_document": True,
            })
            result_id += 1
            if len(results) >= top_k:
                break

        return results

    def build_summary_context(self, filename: Optional[str] = None, max_chunks: int = 10) -> Tuple[str, List[Dict]]:
        """
        Builds a stratified, balanced context digest for summarizing long books and reports.
        Selects TOC / Outlines, Executive Summary, introductory chapters, key tables, and conclusion.
        """
        target_chunks = [c for c in self.chunks if not filename or c["filename"] == filename]
        if not target_chunks:
            return "No document chunks available for summary.", []

        selected = []
        # 1. First 2 chunks (Introduction / Executive Overview)
        selected.extend(target_chunks[:2])

        # 2. Key Table chunks
        table_chunks = [c for c in target_chunks if c.get("is_table") and c not in selected]
        selected.extend(table_chunks[:2])

        # 3. Middle / Chapter chunks distributed evenly
        if len(target_chunks) > 6:
            step = len(target_chunks) // 4
            for i in range(1, 4):
                mid_idx = i * step
                if mid_idx < len(target_chunks) and target_chunks[mid_idx] not in selected:
                    selected.append(target_chunks[mid_idx])

        # 4. Final chunk (Conclusion / Findings)
        if target_chunks[-1] not in selected:
            selected.append(target_chunks[-1])

        # Deduplicate & cap to max_chunks
        unique_selected = []
        seen_ids = set()
        for c in selected:
            if c["chunk_id"] not in seen_ids:
                seen_ids.add(c["chunk_id"])
                unique_selected.append(c)
                if len(unique_selected) >= max_chunks:
                    break

        # Format context
        digest_blocks = []
        ref_sources = []
        for idx, c in enumerate(unique_selected, start=1):
            title = f"{c['filename']} (Page {c['page']})"
            digest_blocks.append(
                f"--- EXCERPT [{idx}] ({title}) ---\n{c['text']}\n"
            )
            ref_sources.append({
                "id": idx,
                "filename": c["filename"],
                "file_type": c["file_type"],
                "page": c["page"],
                "title": title,
                "snippet": c["text"][:260] + "...",
                "is_document": True,
                "is_table": c.get("is_table", False),
                "match_pct": "Summary Reference",
            })

        # Prepend Book TOC if available
        toc_text = ""
        for fn in (self.book_tocs if not filename else [filename]):
            if fn in self.book_tocs:
                toc_lines = [f"• Level {item['level']}: {item['title']} (Page {item['page']})" for item in self.book_tocs[fn][:15]]
                toc_text += f"\n[📖 BOOK OUTLINE / TABLE OF CONTENTS for '{fn}']:\n" + "\n".join(toc_lines) + "\n\n"

        return toc_text + "\n".join(digest_blocks), ref_sources

    def get_summary(self) -> Dict:
        """Returns statistical summary of indexed documents."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "total_tokens": self.total_tokens,
            "total_tables": sum(d.get("tables_count", 0) for d in self.documents.values()),
            "files": list(self.documents.values()),
        }
