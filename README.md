# 🔒 Private AI Search Engine & Book/Tabular Intelligence

A production-ready, privacy-first AI search engine and local document/book assistant built with **Streamlit**, **DuckDuckGo Search**, **PyMuPDF**, **python-docx**, **openpyxl**, and **Ollama / Groq**.

---

## 🌟 Key Capabilities

1. **📖 Book PDF Intelligence**:
   - Handles large 100+ page book PDFs with streaming extraction and memory optimization.
   - Automatically extracts **Table of Contents (TOC) & Chapter Outlines**.
   - Chunks reference the active chapter and exact page: `[contract.pdf • Chapter 2: Terms • Page 14]`.

2. **📊 Structured Tabular Retrieval**:
   - **PDF Tables**: Native PyMuPDF table finder (`page.find_tables()`) converts tables into clean Markdown tables with column headers.
   - **Word Tables**: Parses structured tables from `.docx` files.
   - **Spreadsheets (.xlsx, .xls, .csv)**: Direct ingestion of Excel and CSV sheets.
   - **Tabular Intent Boosting**: Automatically boosts table passages when user queries ask for numbers, metrics, totals, rates, or comparisons.

3. **📋 Executive Stratified Summarization**:
   - Dedicated **"📖 Summarize Book / Doc"** button.
   - Stratified sampling captures TOC, executive summary, key chapters, critical tables, and concluding findings without exceeding LLM context limits.

4. **🎯 Accurate Q&A with Strict Document Grounding**:
   - Answers questions strictly from your uploaded files with inline citations `[1]`, `[2]`.
   - Prevents hallucinations on numerical or tabular questions.

5. **🌐 Untracked Web Search & 🔀 Hybrid Mode**:
   - Private DuckDuckGo web retrieval and hybrid search cross-referencing your documents with live web findings.

---

## 🚀 Quickstart

1. Double-click **`run.bat`** (Windows) or run `python -m streamlit run app.py`.
2. Open **`http://localhost:8501`**.
3. Upload any Book PDF, financial report, or spreadsheet, and query or summarize instantly!
