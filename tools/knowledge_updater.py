"""
academic-research-enhanced â€” KnowledgeUpdater
Self-referential daily crawler: this agent IS the knowledge crawler.
Appends new papers to SECOND-KNOWLEDGE-BRAIN.md at 06:00 daily.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.memory.memory_manager import Paper, ResearchMemoryManager

logger = logging.getLogger("academic_agent.knowledge_updater")

# Daily queries covering the agent's research domains
DAILY_QUERIES = [
    "large language models agents autonomous",
    "retrieval augmented generation RAG knowledge",
    "knowledge graph neural networks embedding",
    "academic paper recommendation system",
    "scientific literature mining NLP",
    "citation network analysis embedding",
    "research gap detection clustering semantic",
    "autonomous research agents AI planning",
    "multi-hop question answering reasoning",
    "survey methodology machine learning systematic",
]

ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.IR", "stat.ML"]

BRAIN_FILE_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "SECOND-KNOWLEDGE-BRAIN.md"
)


class KnowledgeUpdater:
    """Daily self-update pipeline for SECOND-KNOWLEDGE-BRAIN.md."""

    def __init__(
        self,
        memory_manager: Optional["ResearchMemoryManager"] = None,
        brain_file: Optional[str] = None,
    ):
        self.memory_manager = memory_manager
        self.brain_file = brain_file or os.environ.get("BRAIN_FILE", BRAIN_FILE_DEFAULT)

    # ------------------------------------------------------------------
    # Main update entry point
    # ------------------------------------------------------------------

    async def run_daily_update(self) -> Dict[str, Any]:
        """Run the full daily update pipeline."""
        logger.info("Starting daily knowledge base update")
        run_start = datetime.now(timezone.utc)

        all_new_papers = []

        for query in DAILY_QUERIES:
            try:
                papers = await self._crawl_for_query(query)
                all_new_papers.extend(papers)
                await asyncio.sleep(1.0)  # rate limit between queries
            except Exception as e:
                logger.warning("Query '%s' failed: %s", query, e)

        # Deduplicate across all queries
        seen_hashes = set()
        unique_new_papers = []
        for p in all_new_papers:
            h = self._paper_hash(p)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_new_papers.append(p)

        # Score and sort
        for p in unique_new_papers:
            p.score = max(0.1, 1.0 - max(0, datetime.now().year - p.year) * 0.3)

        unique_new_papers.sort(key=lambda p: p.score, reverse=True)

        # Append to SECOND-KNOWLEDGE-BRAIN.md
        brain_updated = False
        if unique_new_papers:
            try:
                self._append_to_brain(unique_new_papers, run_start)
                brain_updated = True
            except Exception as e:
                logger.error("Failed to append to brain file: %s", e)

        # Save to memory manager
        if self.memory_manager and unique_new_papers:
            try:
                self.memory_manager.save_papers(unique_new_papers)
                for p in unique_new_papers:
                    self.memory_manager.mark_paper_known(p.doi, p.title)
            except Exception as e:
                logger.warning("Failed to save papers to memory: %s", e)

        next_run = "tomorrow at 06:00 local time"
        n_added = len(unique_new_papers)

        logger.info("Daily update complete: %d new papers added", n_added)
        print(f"[KnowledgeUpdater] {n_added} new papers added, next run: {next_run}")

        return {
            "papers_added": n_added,
            "next_scheduled_run": next_run,
            "brain_file_updated": brain_updated,
            "run_started": run_start.isoformat(),
            "run_completed": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Crawl per query
    # ------------------------------------------------------------------

    async def _crawl_for_query(self, query: str) -> List[Any]:
        """Crawl ArXiv + Semantic Scholar for a single query (last 3 days)."""
        from agent.modules.paper_crawler import PaperCrawler

        crawler = PaperCrawler(memory_manager=self.memory_manager, rate_limit_delay=1.0)
        try:
            papers = await crawler.crawl(
                query=query,
                sources=["arxiv", "semantic_scholar"],
                max_results=20,
                days_back=3,
            )
            # Filter already-known papers
            if self.memory_manager:
                papers = [p for p in papers if not self.memory_manager.is_known_paper(p.doi)]
            return papers[:20]
        finally:
            await crawler.close()

    # ------------------------------------------------------------------
    # Brain file append
    # ------------------------------------------------------------------

    def _append_to_brain(self, papers: List[Any], run_time: datetime):
        """Append new paper entries to SECOND-KNOWLEDGE-BRAIN.md."""
        date_str = run_time.strftime("%Y-%m-%d")
        n = len(papers)

        lines = [
            f"\n### [{date_str}] Daily Update -- {n} papers added\n",
            "| Title | Authors | Year | Source | URL | Key Finding |\n",
            "|-------|---------|------|--------|-----|-------------|\n",
        ]

        for p in papers:
            authors_str = ", ".join(p.authors[:2])
            if len(p.authors) > 2:
                authors_str += " et al."
            title_short = p.title[:80] + "..." if len(p.title) > 80 else p.title
            abstract_short = p.abstract[:120] + "..." if len(p.abstract) > 120 else p.abstract
            url = p.url or ""
            # Sanitize for Markdown table (no pipes inside cells)
            title_md = title_short.replace("|", "-")
            abstract_md = abstract_short.replace("|", "-").replace("\n", " ")

            lines.append(
                f"| {title_md} | {authors_str} | {p.year} | {p.source} | {url} | {abstract_md} |\n"
            )

        # Append to the brain file
        os.makedirs(os.path.dirname(self.brain_file), exist_ok=True) if os.path.dirname(self.brain_file) else None

        with open(self.brain_file, "a", encoding="utf-8") as fh:
            fh.writelines(lines)

        logger.info("Appended %d new papers to %s", n, self.brain_file)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _paper_hash(self, paper: Any) -> str:
        key = f"{paper.doi.lower().strip()}|{paper.title.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Schedule (standalone runner)
    # ------------------------------------------------------------------

    def start_scheduled(self):
        """Start APScheduler for daily cron at 06:00 (standalone usage)."""
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger

            scheduler = AsyncIOScheduler()
            scheduler.add_job(
                self.run_daily_update,
                trigger=CronTrigger(hour=6, minute=0),
                id="knowledge_daily_update",
                replace_existing=True,
                max_instances=1,
            )
            scheduler.start()
            logger.info("KnowledgeUpdater scheduled at 06:00 daily")
            return scheduler
        except ImportError:
            logger.warning("APScheduler not available; cannot start scheduled updates")
            return None


# ------------------------------------------------------------------
# CLI entry point for standalone use
# ------------------------------------------------------------------

async def _main():
    """Run a single knowledge update pass."""
    from agent.memory.memory_manager import Paper, ResearchMemoryManager
    mm = ResearchMemoryManager()
    updater = KnowledgeUpdater(memory_manager=mm)
    result = await updater.run_daily_update()
    print(f"Done: {result}")


if __name__ == "__main__":
    asyncio.run(_main())
