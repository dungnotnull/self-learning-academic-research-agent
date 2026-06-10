"""
academic-research-enhanced — PaperCrawler
Multi-source academic paper retrieval: ArXiv, Semantic Scholar, PubMed, SSRN.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, TYPE_CHECKING
from xml.etree import ElementTree as ET

import aiohttp

from agent.memory.memory_manager import Paper

if TYPE_CHECKING:
    from agent.memory.memory_manager import ResearchMemoryManager

logger = logging.getLogger("academic_agent.paper_crawler")


class PaperCrawler:
    """Async multi-source paper crawler with deduplication and scoring."""

    ARXIV_BASE = "http://export.arxiv.org/api/query"
    S2_BASE = "https://api.semanticscholar.org/graph/v1"
    PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    SSRN_SEARCH = "https://www.ssrn.com/index.cfm/en/eLibrary/"

    ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.IR", "stat.ML"]

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None, rate_limit_delay: float = 1.0):
        self.memory_manager = memory_manager
        self.rate_limit_delay = rate_limit_delay
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def crawl(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: int = 50,
        days_back: int = 7,
    ) -> List[Paper]:
        """Crawl all enabled sources, deduplicate, score, and return sorted papers."""
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]

        all_papers: List[Paper] = []
        tasks = []

        if "arxiv" in sources:
            tasks.append(self._crawl_arxiv(query, max_results, days_back))
        if "semantic_scholar" in sources:
            tasks.append(self._crawl_semantic_scholar(query, max_results, days_back))
        if "pubmed" in sources:
            tasks.append(self._crawl_pubmed(query, min(max_results, 20), days_back))
        if "ssrn" in sources:
            tasks.append(self._crawl_ssrn(query, min(max_results, 20)))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Source crawl failed: %s", r)
            elif isinstance(r, list):
                all_papers.extend(r)

        # Deduplicate
        unique_papers = self._deduplicate(all_papers)

        # Score
        for p in unique_papers:
            p.score = self._score_paper(p, query, days_back)

        # Sort by score descending
        unique_papers.sort(key=lambda p: p.score, reverse=True)

        logger.info(
            "Crawled %d papers (%d unique) for query: %s",
            len(all_papers),
            len(unique_papers),
            query,
        )
        return unique_papers

    # ------------------------------------------------------------------
    # ArXiv
    # ------------------------------------------------------------------

    async def _crawl_arxiv(self, query: str, max_results: int, days_back: int) -> List[Paper]:
        """Crawl ArXiv XML API."""
        papers: List[Paper] = []
        session = await self._get_session()

        categories_filter = " OR ".join(f"cat:{c}" for c in self.ARXIV_CATEGORIES)
        search_query = f"({query}) AND ({categories_filter})"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        try:
            await asyncio.sleep(self.rate_limit_delay)
            async with session.get(self.ARXIV_BASE, params=params) as resp:
                resp.raise_for_status()
                xml_text = await resp.text()

            root = ET.fromstring(xml_text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

            for entry in root.findall("atom:entry", ns):
                try:
                    paper = self._parse_arxiv_entry(entry, ns, cutoff)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.debug("Failed to parse ArXiv entry: %s", e)

        except Exception as e:
            logger.warning("ArXiv crawl failed: %s", e)

        logger.debug("ArXiv returned %d papers", len(papers))
        return papers

    def _parse_arxiv_entry(self, entry, ns: dict, cutoff: datetime) -> Optional[Paper]:
        """Parse a single ArXiv Atom entry into a Paper."""
        title_el = entry.find("atom:title", ns)
        abstract_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        id_el = entry.find("atom:id", ns)

        if title_el is None or abstract_el is None:
            return None

        title = re.sub(r"\s+", " ", title_el.text or "").strip()
        abstract = re.sub(r"\s+", " ", abstract_el.text or "").strip()

        pub_date_str = published_el.text if published_el is not None else ""
        try:
            pub_dt = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
        except Exception:
            pub_dt = datetime.now(timezone.utc)

        if pub_dt < cutoff:
            return None

        arxiv_url = id_el.text.strip() if id_el is not None else ""
        arxiv_id = arxiv_url.split("/abs/")[-1] if "/abs/" in arxiv_url else ""

        authors = []
        for author_el in entry.findall("atom:author", ns):
            name_el = author_el.find("atom:name", ns)
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        doi_el = entry.find("arxiv:doi", ns)
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else f"arxiv:{arxiv_id}"

        primary_cat = entry.find("arxiv:primary_category", ns)
        venue = primary_cat.get("term", "arXiv") if primary_cat is not None else "arXiv"

        return Paper(
            doi=doi,
            title=title,
            authors=authors,
            abstract=abstract,
            year=pub_dt.year,
            venue=venue,
            url=arxiv_url,
            source="arxiv",
            arxiv_id=arxiv_id,
        )

    # ------------------------------------------------------------------
    # Semantic Scholar
    # ------------------------------------------------------------------

    async def _crawl_semantic_scholar(self, query: str, max_results: int, days_back: int) -> List[Paper]:
        """Crawl Semantic Scholar Graph API."""
        papers: List[Paper] = []
        session = await self._get_session()

        fields = "title,authors,year,citationCount,externalIds,abstract,url,venue"
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": fields,
        }

        import os
        s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        headers = {"x-api-key": s2_key} if s2_key else {}

        cutoff_year = (datetime.now(timezone.utc) - timedelta(days=days_back)).year

        try:
            await asyncio.sleep(self.rate_limit_delay)
            async with session.get(
                f"{self.S2_BASE}/paper/search",
                params=params,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()

            for item in data.get("data", []):
                try:
                    year = item.get("year") or 0
                    if year < cutoff_year - 1:
                        continue

                    ext_ids = item.get("externalIds") or {}
                    doi = (
                        ext_ids.get("DOI")
                        or (f"arxiv:{ext_ids['ArXiv']}" if ext_ids.get("ArXiv") else "")
                        or item.get("paperId", "")
                    )
                    if not doi:
                        doi = f"s2:{item.get('paperId', hashlib.md5(item.get('title', '').encode()).hexdigest())}"

                    authors = [a.get("name", "") for a in (item.get("authors") or [])]

                    papers.append(Paper(
                        doi=str(doi),
                        title=item.get("title") or "",
                        authors=authors,
                        abstract=item.get("abstract") or "",
                        year=year,
                        venue=item.get("venue") or "Semantic Scholar",
                        url=item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId', '')}",
                        source="semantic_scholar",
                        citation_count=item.get("citationCount") or 0,
                        arxiv_id=ext_ids.get("ArXiv"),
                    ))
                except Exception as e:
                    logger.debug("Failed to parse S2 entry: %s", e)

        except Exception as e:
            logger.warning("Semantic Scholar crawl failed: %s", e)

        logger.debug("Semantic Scholar returned %d papers", len(papers))
        return papers

    # ------------------------------------------------------------------
    # PubMed
    # ------------------------------------------------------------------

    async def _crawl_pubmed(self, query: str, max_results: int, days_back: int) -> List[Paper]:
        """Crawl PubMed via NCBI E-utilities."""
        papers: List[Paper] = []
        session = await self._get_session()

        import os
        ncbi_key = os.environ.get("NCBI_API_KEY", "")
        ncbi_email = os.environ.get("NCBI_EMAIL", "researcher@example.com")

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y/%m/%d")
        today = datetime.now(timezone.utc).strftime("%Y/%m/%d")

        esearch_params = {
            "db": "pubmed",
            "term": f"{query}[Title/Abstract]",
            "retmax": max_results,
            "retmode": "json",
            "datetype": "pdat",
            "mindate": cutoff,
            "maxdate": today,
            "email": ncbi_email,
        }
        if ncbi_key:
            esearch_params["api_key"] = ncbi_key

        try:
            await asyncio.sleep(self.rate_limit_delay)
            async with session.get(f"{self.PUBMED_BASE}/esearch.fcgi", params=esearch_params) as resp:
                resp.raise_for_status()
                search_data = await resp.json()

            pmids = search_data.get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []

            await asyncio.sleep(self.rate_limit_delay)
            efetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract",
                "email": ncbi_email,
            }
            if ncbi_key:
                efetch_params["api_key"] = ncbi_key

            async with session.get(f"{self.PUBMED_BASE}/efetch.fcgi", params=efetch_params) as resp:
                resp.raise_for_status()
                xml_text = await resp.text()

            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                try:
                    paper = self._parse_pubmed_article(article)
                    if paper:
                        papers.append(paper)
                except Exception as e:
                    logger.debug("Failed to parse PubMed article: %s", e)

        except Exception as e:
            logger.warning("PubMed crawl failed: %s", e)

        logger.debug("PubMed returned %d papers", len(papers))
        return papers

    def _parse_pubmed_article(self, article) -> Optional[Paper]:
        """Parse PubMed XML article element."""
        citation = article.find("MedlineCitation")
        if citation is None:
            return None

        pmid_el = citation.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        article_el = citation.find("Article")
        if article_el is None:
            return None

        title_el = article_el.find("ArticleTitle")
        title = title_el.text or "" if title_el is not None else ""

        abstract_text_parts = []
        for abstract_el in article_el.findall(".//AbstractText"):
            if abstract_el.text:
                abstract_text_parts.append(abstract_el.text)
        abstract = " ".join(abstract_text_parts)

        year = 0
        journal_el = article_el.find("Journal/JournalIssue/PubDate")
        if journal_el is not None:
            year_el = journal_el.find("Year")
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text)
                except ValueError:
                    year = datetime.now().year

        authors = []
        for author in article_el.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                authors.append(f"{fore} {last}".strip())

        doi = f"pubmed:{pmid}"
        for id_el in article.findall(".//ArticleId"):
            if id_el.get("IdType") == "doi" and id_el.text:
                doi = id_el.text.strip()
                break

        journal_title = article_el.findtext("Journal/Title", "PubMed")

        return Paper(
            doi=doi,
            title=title.strip(),
            authors=authors,
            abstract=abstract.strip(),
            year=year,
            venue=journal_title,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            source="pubmed",
        )

    # ------------------------------------------------------------------
    # SSRN
    # ------------------------------------------------------------------

    async def _crawl_ssrn(self, query: str, max_results: int) -> List[Paper]:
        """Crawl SSRN via HTML parsing (no official API)."""
        papers: List[Paper] = []
        session = await self._get_session()

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4 not installed; skipping SSRN crawl")
            return []

        try:
            await asyncio.sleep(self.rate_limit_delay * 2)
            async with session.get(
                "https://papers.ssrn.com/sol3/results.cfm",
                params={"form_name": "journalbrowse", "strSearch": query},
                headers={"User-Agent": "AcademicResearchAgent/1.0 (research)"},
            ) as resp:
                if resp.status != 200:
                    logger.debug("SSRN returned status %d", resp.status)
                    return []
                html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            paper_divs = soup.find_all("div", class_="title-view-right")[:max_results]

            for div in paper_divs:
                try:
                    link = div.find("a")
                    if not link:
                        continue
                    title = link.text.strip()
                    url = "https://papers.ssrn.com" + link.get("href", "")
                    doi = f"ssrn:{hashlib.md5(url.encode()).hexdigest()[:12]}"

                    authors_div = div.find_next_sibling("div", class_="authors")
                    authors_text = authors_div.text.strip() if authors_div else ""
                    authors = [a.strip() for a in authors_text.split(",")]

                    papers.append(Paper(
                        doi=doi,
                        title=title,
                        authors=authors,
                        abstract="",
                        year=datetime.now().year,
                        venue="SSRN",
                        url=url,
                        source="ssrn",
                    ))
                except Exception as e:
                    logger.debug("Failed to parse SSRN entry: %s", e)

        except Exception as e:
            logger.warning("SSRN crawl failed: %s", e)

        logger.debug("SSRN returned %d papers", len(papers))
        return papers

    # ------------------------------------------------------------------
    # Deduplication and scoring
    # ------------------------------------------------------------------

    def _paper_hash(self, paper: Paper) -> str:
        """Compute SHA256 hash from doi+title for deduplication."""
        key = f"{paper.doi.lower().strip()}|{paper.title.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()

    def _deduplicate(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers; prefer higher citation count version."""
        seen: dict = {}
        for p in papers:
            h = self._paper_hash(p)
            if h not in seen:
                seen[h] = p
            else:
                existing = seen[h]
                if p.citation_count > existing.citation_count or (not existing.abstract and p.abstract):
                    seen[h] = p

        # Also filter papers already in MemoryManager
        unique = list(seen.values())
        if self.memory_manager:
            unique = [p for p in unique if not self.memory_manager.is_known_paper(p.doi)]

        return unique

    def _score_paper(self, paper: Paper, query: str, days_back: int) -> float:
        """Compute recency x relevance score (0..1)."""
        now_year = datetime.now().year
        age_years = max(0, now_year - paper.year)
        recency_score = max(0.1, 1.0 - (age_years / max(days_back / 365, 1)))

        text = f"{paper.title} {paper.abstract}".lower()
        query_terms = re.findall(r"\w+", query.lower())
        if not query_terms:
            relevance_score = 0.5
        else:
            matched = sum(1 for t in query_terms if t in text)
            relevance_score = matched / len(query_terms)

        return recency_score * 0.6 + relevance_score * 0.4

    def _save_papers(self, papers: List[Paper]):
        """Batch save papers to MemoryManager."""
        if self.memory_manager:
            self.memory_manager.save_papers(papers)
