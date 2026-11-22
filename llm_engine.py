"""
llm_engine.py - High-performance LLM orchestrator supporting local Ollama & cloud Groq inference.
Includes unified support for Web Search, Local Document RAG, Tabular Data Reasoning,
Book Summarization, and Hybrid search, with citation prompting and real-time token streaming.
"""

import logging
from typing import Dict, Generator, List, Optional

logger = logging.getLogger("LLMEngine")

SYSTEM_PROMPT_WEB = """You are an advanced, privacy-first AI Research Assistant.
Your objective is to provide an accurate, insightful, and well-structured answer to the user's query based strictly on the provided web search context.

STRICT GUIDELINES:
1. CITATIONS: Cite every factual claim, statistic, or piece of information using bracketed numeric markers like [1], [2], or [1, 3] directly after the corresponding sentence or phrase.
2. CITATION CONSISTENCY: Only use citation indices that exist in the provided sources (e.g., [1] through [{max_source_idx}]).
3. OBJECTIVITY & TRUTHFULNESS: Do not fabricate facts. If the provided sources are insufficient to fully answer the query, explicitly state what is known and mention what information is missing.
4. FORMATTING: Structure your response cleanly using markdown (concise paragraphs, bullet points, headers, bold keywords).
5. CLEAN FOOTER: Do NOT print a manual bibliography or URLs list at the end of your response. The application UI automatically generates interactive visual source cards from your [X] citations."""

SYSTEM_PROMPT_DOCS = """You are an advanced, privacy-first AI Document Research Assistant.
Your objective is to provide an accurate, insightful, and well-structured answer to the user's query based STRICTLY on the uploaded local document excerpts (Books, PDFs, Tabular Datasets, Word DOCX, or Images).

STRICT GUIDELINES:
1. CITATIONS: Cite every factual claim, number, or piece of information using bracketed numeric markers like [1], [2], or [1, 3] directly after the corresponding sentence or phrase.
2. TABULAR PRECISION: When answering questions regarding tables, financial figures, metrics, or comparisons, extract exact numbers and format them in clear Markdown tables or bullet lists.
3. CITATION CONSISTENCY: Only use citation indices that exist in the provided sources ([1] through [{max_source_idx}]).
4. ATTRIBUTION: Cite specific pages and tables (e.g., "According to Table 1 on Page 4 of report.pdf [1]...").
5. OBJECTIVITY & TRUTHFULNESS: Do not fabricate facts or extrapolate beyond what is in the documents. If the document excerpts do not contain the answer, explicitly state: "The uploaded document does not provide details on [topic]."
6. FORMATTING: Use markdown headings, bullet points, and clean tables for readability.
7. CLEAN FOOTER: Do NOT print a manual bibliography at the end."""

SYSTEM_PROMPT_SUMMARY = """You are an expert executive document analyst and book researcher.
Your task is to generate a comprehensive, structured, and insightful summary of the provided document / book excerpts and outline.

STRUCTURE YOUR SUMMARY CLEANLY:
1. 📌 Executive Overview: Core thesis, objective, or primary subject matter of the document.
2. 📖 Book Structure & Chapter Roadmap: Key topics, sections, or chapter progression.
3. 📊 Key Data, Metrics & Tabular Highlights: Prominent quantitative figures, statistics, financial data, or key table findings.
4. 💡 Critical Insights & Strategic Takeaways: The most significant conclusions, recommendations, and actionable takeaways.

GUIDELINES:
- Cite source excerpts using [1], [2] where specific points or data originate.
- Format tabular data using clean Markdown tables.
- Keep the analysis factual, sharp, and easy to read."""

SYSTEM_PROMPT_HYBRID = """You are an advanced, privacy-first AI Research Assistant with access to BOTH the user's uploaded local documents AND live untracked web search.
Your objective is to provide an accurate, well-synthesized answer that cross-references the user's documents with the latest web findings.

STRICT GUIDELINES:
1. CITATIONS: Cite every claim using bracketed numeric markers [1], [2], etc., corresponding to either the document sources or web sources.
2. DISTINCTION: Clearly distinguish between findings from the user's local documents versus findings from live web search.
3. TABULAR DATA: Present comparative numbers in clean Markdown tables.
4. CLEAN FOOTER: Do NOT print a manual bibliography list at the end."""


class LLMManager:
    """Manages chat models (Groq / Ollama) and streaming answer synthesis."""

    def __init__(
        self,
        provider: str = "ollama",
        model_name: str = "llama3.2:latest",
        temperature: float = 0.2,
        groq_api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.groq_api_key = groq_api_key
        self.ollama_base_url = ollama_base_url
        self.backend_type = None  # 'langchain', 'native_groq', or 'native_ollama'
        self.client = self._init_model()

    def _init_model(self):
        """Initializes the LLM backend via LangChain or native SDK fallback."""
        if self.provider == "groq":
            if not self.groq_api_key:
                raise ValueError("Groq API Key is required when using the Groq provider.")

            try:
                from langchain_groq import ChatGroq
                self.backend_type = "langchain"
                return ChatGroq(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    groq_api_key=self.groq_api_key,
                    streaming=True,
                )
            except Exception:
                from groq import Groq
                self.backend_type = "native_groq"
                return Groq(api_key=self.groq_api_key)

        elif self.provider == "ollama":
            try:
                try:
                    from langchain_ollama import ChatOllama
                except ImportError:
                    from langchain_community.chat_models import ChatOllama

                self.backend_type = "langchain"
                return ChatOllama(
                    model=self.model_name,
                    base_url=self.ollama_base_url,
                    temperature=self.temperature,
                    streaming=True,
                )
            except Exception:
                import ollama
                self.backend_type = "native_ollama"
                return ollama.Client(host=self.ollama_base_url)

        else:
            raise ValueError(f"Unsupported provider: '{self.provider}'. Choose 'groq' or 'ollama'.")

    @staticmethod
    def format_search_context(sources: List[Dict], mode: str = "web") -> str:
        """Formats list of retrieved sources (web or local documents) into numbered context blocks."""
        if not sources:
            return "No search results or document excerpts available."

        formatted_blocks = []
        for src in sources:
            is_doc = src.get("is_document", False)
            if is_doc:
                chapter_str = f" | Section: {src.get('chapter')}" if src.get("chapter") and src.get("chapter") != "General" else ""
                block = (
                    f"--- DOCUMENT SOURCE [{src['id']}] ---\n"
                    f"File: {src.get('filename', 'document')}\n"
                    f"Format: {src.get('file_type', 'file').upper()}\n"
                    f"Location: Page {src.get('page', 1)}{chapter_str}\n"
                    f"Content Excerpt:\n{src.get('content', src.get('snippet', ''))}\n"
                )
            else:
                block = (
                    f"--- WEB SOURCE [{src['id']}] ---\n"
                    f"Title: {src.get('title', 'Web Result')}\n"
                    f"Domain: {src.get('domain', 'web')}\n"
                    f"URL: {src.get('url', '')}\n"
                    f"Content Excerpt:\n{src.get('content', src.get('snippet', ''))}\n"
                )
            formatted_blocks.append(block)

        return "\n".join(formatted_blocks)

    def stream_answer(
        self,
        query: str,
        sources: List[Dict],
        mode: str = "web",
        chat_history: Optional[List[Dict]] = None,
        custom_context: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Streams synthesized response chunks."""
        context_str = custom_context or self.format_search_context(sources, mode=mode)
        max_idx = len(sources) if sources else 1

        if mode == "summary":
            template = SYSTEM_PROMPT_SUMMARY
            user_content = (
                f"DOCUMENT EXCERPTS & OUTLINE:\n{context_str}\n\n"
                f"Generate a comprehensive, structured, executive-level summary and analysis."
            )
        elif mode == "document":
            template = SYSTEM_PROMPT_DOCS
            user_content = (
                f"USER QUESTION:\n{query}\n\n"
                f"DOCUMENT CONTEXT (INCLUDING TABLES & TEXT):\n{context_str}\n\n"
                f"Answer the question directly and precisely, citing the document name, page, and table using [1], [2], etc."
            )
        elif mode == "hybrid":
            template = SYSTEM_PROMPT_HYBRID
            user_content = (
                f"USER QUESTION:\n{query}\n\n"
                f"COMBINED DOCUMENT & WEB CONTEXT:\n{context_str}\n\n"
                f"Synthesize an answer utilizing both the local document excerpts and web information with [1], [2] citations."
            )
        else:
            template = SYSTEM_PROMPT_WEB
            user_content = (
                f"USER QUERY:\n{query}\n\n"
                f"WEB SEARCH CONTEXT:\n{context_str}\n\n"
                f"Synthesize a clear, direct, and well-cited response based strictly on the above sources."
            )

        system_instruction = template.format(max_source_idx=max_idx)

        # 1. LangChain backend
        if self.backend_type == "langchain":
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            messages = [SystemMessage(content=system_instruction)]
            if chat_history and mode != "summary":
                for msg in chat_history[-4:]:
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        messages.append(AIMessage(content=msg["content"]))

            messages.append(HumanMessage(content=user_content))

            try:
                for chunk in self.client.stream(messages):
                    yield chunk.content
            except Exception as e:
                logger.error(f"Error during LangChain streaming: {e}")
                yield f"\n\n**Error during inference:** {str(e)}"

        # 2. Native Groq backend
        elif self.backend_type == "native_groq":
            native_messages = [{"role": "system", "content": system_instruction}]
            if chat_history and mode != "summary":
                for msg in chat_history[-4:]:
                    if msg.get("role") in ("user", "assistant"):
                        native_messages.append({"role": msg["role"], "content": msg["content"]})
            native_messages.append({"role": "user", "content": user_content})

            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=native_messages,
                    temperature=self.temperature,
                    stream=True,
                )
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Error during Groq streaming: {e}")
                yield f"\n\n**Error during inference:** {str(e)}"

        # 3. Native Ollama backend
        elif self.backend_type == "native_ollama":
            native_messages = [{"role": "system", "content": system_instruction}]
            if chat_history and mode != "summary":
                for msg in chat_history[-4:]:
                    if msg.get("role") in ("user", "assistant"):
                        native_messages.append({"role": msg["role"], "content": msg["content"]})
            native_messages.append({"role": "user", "content": user_content})

            try:
                stream = self.client.chat(
                    model=self.model_name,
                    messages=native_messages,
                    stream=True,
                    options={"temperature": self.temperature},
                )
                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
            except Exception as e:
                logger.error(f"Error during Ollama streaming: {e}")
                yield (
                    f"\n\n**Ollama Connection Error:** {str(e)}\n"
                    f"Ensure Ollama is running and model '{self.model_name}' is downloaded (`ollama pull {self.model_name}`)."
                )
