"""
search_engine.py - Untracked, Private Web Search & Content Extraction Engine.
Features multi-layer fallback:
1. `ddgs` (new official DuckDuckGo Python engine)
2. `duckduckgo_search` (legacy fallback)
3. Direct HTML privacy endpoint fallback
4. Concurrent webpage scraper for rich context
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse
import requests
from bs4 import BeautifulSoup

# Try importing ddgs first, fallback to duckduckgo_search
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

logger = logging.getLogger("PrivateSearchEngine")


class PrivateSearchEngine:
    """Handles private web search and content extraction without logging or user tracking."""

    def __init__(self, searxng_url: Optional[str] = None, timeout: float = 3.5):
        self.searxng_url = searxng_url.rstrip("/") if searxng_url else None
        self.scrape_timeout = timeout
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

    def _extract_domain(self, url: str) -> str:
        """Extracts clean domain name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain if domain else "web"
        except Exception:
            return "web"

    def _search_ddg_library(self, query: str, max_results: int = 5) -> List[Dict]:
        """Performs search via DDGS library."""
        if not DDGS:
            return []
        results = []
        try:
            with DDGS() as ddgs:
                ddg_gen = ddgs.text(query, max_results=max_results)
                for i, r in enumerate(ddg_gen, start=1):
                    url = r.get("href") or r.get("link") or ""
                    title = r.get("title", "Untitled Source").strip()
                    snippet = (r.get("body") or r.get("snippet") or "").strip()
                    if url and title:
                        results.append({
                            "id": i,
                            "title": title,
                            "url": url,
                            "domain": self._extract_domain(url),
                            "snippet": snippet,
                            "content": snippet,
                            "scraped": False,
                        })
        except Exception as e:
            logger.warning(f"DDGS library search failed: {e}")
        return results

    def _search_ddg_html_fallback(self, query: str, max_results: int = 5) -> List[Dict]:
        """Fallback search using direct DuckDuckGo HTML endpoint."""
        results = []
        try:
            endpoint = "https://html.duckduckgo.com/html/"
            data = {"q": query, "b": ""}
            resp = requests.post(endpoint, data=data, headers=self.headers, timeout=6.0)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                blocks = soup.find_all("div", class_="result")
                idx = 1
                for block in blocks:
                    title_elem = block.find("a", class_="result__a")
                    snippet_elem = block.find("a", class_="result__snippet")
                    if not title_elem:
                        continue

                    raw_href = title_elem.get("href", "")
                    # Extract direct destination if wrapped in DuckDuckGo redirect
                    if "uddg=" in raw_href:
                        actual_url = unquote(raw_href.split("uddg=")[-1].split("&")[0])
                    else:
                        actual_url = raw_href

                    title = title_elem.get_text(strip=True)
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    if actual_url and title:
                        results.append({
                            "id": idx,
                            "title": title,
                            "url": actual_url,
                            "domain": self._extract_domain(actual_url),
                            "snippet": snippet,
                            "content": snippet,
                            "scraped": False,
                        })
                        idx += 1
                        if len(results) >= max_results:
                            break
        except Exception as e:
            logger.error(f"DDG HTML fallback error: {e}")
        return results

    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict]:
        """Performs private search with automatic fallback."""
        results = self._search_ddg_library(query, max_results=max_results)
        if not results:
            logger.info("DDGS library returned empty; trying direct DDG HTML fallback...")
            results = self._search_ddg_html_fallback(query, max_results=max_results)
        return results

    def search_searxng(self, query: str, max_results: int = 5) -> List[Dict]:
        """Performs search via self-hosted SearXNG instance."""
        if not self.searxng_url:
            return []
        results = []
        try:
            endpoint = f"{self.searxng_url}/search"
            params = {
                "q": query,
                "format": "json",
                "categories": "general",
                "language": "auto",
            }
            resp = requests.get(endpoint, params=params, headers=self.headers, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                raw_results = data.get("results", [])[:max_results]
                for i, r in enumerate(raw_results, start=1):
                    url = r.get("url", "")
                    snippet = r.get("content", "").strip()
                    results.append({
                        "id": i,
                        "title": r.get("title", "Untitled Source"),
                        "url": url,
                        "domain": self._extract_domain(url),
                        "snippet": snippet,
                        "content": snippet,
                        "scraped": False,
                    })
        except Exception as e:
            logger.error(f"SearXNG search error: {e}")
        return results

    def _scrape_page(self, item: Dict, max_chars: int = 1500) -> Dict:
        """Fetches page HTML and extracts clean text; falls back to snippet on failure."""
        url = item.get("url", "")
        if not url or url.startswith("#"):
            return item

        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.scrape_timeout,
                allow_redirects=True,
            )
            if response.status_code == 200 and "text/html" in response.headers.get("Content-Type", ""):
                soup = BeautifulSoup(response.text, "html.parser")

                for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "form"]):
                    element.decompose()

                paragraphs = soup.find_all(["p", "h1", "h2", "h3", "li"])
                text_parts = [p.get_text(separator=" ", strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 25]
                clean_text = " ".join(text_parts)

                if len(clean_text) >= 100:
                    item["content"] = clean_text[:max_chars]
                    item["scraped"] = True
        except Exception as e:
            logger.debug(f"Failed scraping {url}: {e}")

        return item

    def search_and_enrich(
        self,
        query: str,
        backend: str = "duckduckgo",
        max_results: int = 5,
        deep_scrape: bool = True,
    ) -> List[Dict]:
        """
        Executes web search and concurrently deep-scrapes target webpages for rich context.
        """
        if backend == "searxng" and self.searxng_url:
            results = self.search_searxng(query, max_results=max_results)
            if not results:
                results = self.search_duckduckgo(query, max_results=max_results)
        else:
            results = self.search_duckduckgo(query, max_results=max_results)

        if not results:
            return []

        if not deep_scrape:
            return results

        # Deep scrape in parallel with ThreadPoolExecutor
        enriched_results = []
        with ThreadPoolExecutor(max_workers=min(len(results), 5)) as executor:
            future_to_item = {executor.submit(self._scrape_page, item): item for item in results}
            for future in as_completed(future_to_item):
                try:
                    enriched = future.result()
                    enriched_results.append(enriched)
                except Exception:
                    enriched_results.append(future_to_item[future])

        enriched_results.sort(key=lambda x: x["id"])
        return enriched_results
