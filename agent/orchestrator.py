"""
academic-research-enhanced — ResearchOrchestrator
Core agent decision loop. Wires all modules together.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("academic_agent.orchestrator")

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    _SCHEDULER_AVAILABLE = True
except ImportError:
    _SCHEDULER_AVAILABLE = False
    logger.warning("APScheduler not available; daily self-update cron disabled")

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server as _prom_start
    papers_crawled_total = Counter("papers_crawled_total", "Total papers ingested")
    syntheses_generated_total = Counter("syntheses_generated_total", "Total synthesis calls")
    gaps_found_total = Counter("gaps_found_total", "Total research gaps identified")
    llm_tokens_used_total = Counter("llm_tokens_used_total", "Total LLM tokens used", ["provider"])
    llm_call_latency_seconds = Histogram(
        "llm_call_latency_seconds",
        "LLM call latency in seconds",
        ["provider", "task"],
        buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


class ResearchOrchestrator:
    """Central orchestrator for the academic research agent."""

    VERSION = "1.0.0"

    def __init__(self, config_path: Optional[str] = None):
        from agent.memory.memory_manager import ResearchMemoryManager
        self.memory_manager = ResearchMemoryManager()
        self._config = self._load_config(config_path)
        self._crawler = None
        self._citation_analyzer = None
        self._gap_finder = None
        self._synthesis_engine = None
        self._scheduler = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, config_path: Optional[str]) -> dict:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "agent_config.yaml"
            )
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        except Exception:
            logger.warning("Could not load agent_config.yaml; using defaults")
            return {}

    # ------------------------------------------------------------------
    # Lazy module initialization
    # ------------------------------------------------------------------

    def _get_crawler(self):
        if self._crawler is None:
            from agent.modules.paper_crawler import PaperCrawler
            self._crawler = PaperCrawler(memory_manager=self.memory_manager)
        return self._crawler

    def _get_citation_analyzer(self):
        if self._citation_analyzer is None:
            from agent.modules.citation_analyzer import CitationAnalyzer
            self._citation_analyzer = CitationAnalyzer(memory_manager=self.memory_manager)
        return self._citation_analyzer

    def _get_gap_finder(self):
        if self._gap_finder is None:
            from agent.modules.research_gap_finder import ResearchGapFinder
            self._gap_finder = ResearchGapFinder(memory_manager=self.memory_manager)
        return self._gap_finder

    def _get_synthesis_engine(self):
        if self._synthesis_engine is None:
            from agent.modules.synthesis_engine import SynthesisEngine
            self._synthesis_engine = SynthesisEngine(memory_manager=self.memory_manager)
        return self._synthesis_engine

    # ------------------------------------------------------------------
    # Core async pipeline methods
    # ------------------------------------------------------------------

    async def run_full_pipeline(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_papers: int = 50,
    ) -> Dict[str, Any]:
        """Crawl -> analyze -> find gaps -> synthesize -> update KB."""
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]

        logger.info("Starting full pipeline for query: %s", query)
        pipeline_start = time.time()

        # Step 1: Crawl
        crawl_result = await self.crawl_papers(
            query=query, sources=sources, max_results=max_papers, days_back=7
        )

        # Step 2: Citation analysis
        analyze_result = await self.analyze_citations(topic=query, min_citations=3)

        # Step 3: Gap detection
        gap_result = await self.find_gaps(topic=query)

        # Step 4: Synthesis
        synthesis_result = await self.synthesize_literature(
            query=query, max_papers=min(max_papers, 15), style="academic"
        )

        # Step 5: Update KB
        kb_result = await self.update_knowledge_base()

        elapsed = time.time() - pipeline_start

        report = {
            "query": query,
            "elapsed_seconds": round(elapsed, 2),
            "crawl": crawl_result,
            "citation_analysis": analyze_result,
            "gaps": gap_result,
            "synthesis": synthesis_result,
            "knowledge_base": kb_result,
            "pipeline_completed_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "Full pipeline complete in %.1fs: %d papers, %d gaps, review quality %.2f",
            elapsed,
            crawl_result.get("papers_found", 0),
            gap_result.get("gaps_found", 0),
            synthesis_result.get("quality_score", 0),
        )

        return report

    async def crawl_papers(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_results: int = 50,
        days_back: int = 7,
    ) -> Dict[str, Any]:
        """Delegate to PaperCrawler and return summary dict."""
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]

        crawler = self._get_crawler()
        papers = await crawler.crawl(
            query=query,
            sources=sources,
            max_results=max_results,
            days_back=days_back,
        )

        new_count = sum(1 for p in papers if not self.memory_manager.is_known_paper(p.doi))
        if new_count > 0:
            self.memory_manager.save_papers(papers)
            for p in papers:
                if not self.memory_manager.is_known_paper(p.doi):
                    self.memory_manager.mark_paper_known(p.doi, p.title)

        if _PROMETHEUS_AVAILABLE:
            papers_crawled_total.inc(len(papers))

        sample_titles = [p.title for p in papers[:5]]
        return {
            "query": query,
            "papers_found": len(papers),
            "papers_new": new_count,
            "sources_queried": sources,
            "sample_titles": sample_titles,
        }

    async def analyze_citations(
        self,
        topic: str,
        min_citations: int = 5,
    ) -> Dict[str, Any]:
        """Build citation graph and return influence metrics."""
        papers = self.memory_manager.search_papers(topic, limit=200)
        if len(papers) < 5:
            logger.warning("Too few papers (%d) for citation analysis of topic: %s", len(papers), topic)
            return {
                "topic": topic,
                "graph_nodes": 0,
                "graph_edges": 0,
                "top_influential": [],
                "bridge_papers": [],
                "warning": "Fewer than 5 papers available; crawl more papers first",
            }

        analyzer = self._get_citation_analyzer()
        graph = await analyzer.build_citation_graph(papers)
        influential = analyzer.identify_influential(graph, top_k=10)
        bridges = analyzer.identify_bridge_papers(graph, top_k=5)

        edge_count = graph.number_of_edges()
        node_count = graph.number_of_nodes()

        return {
            "topic": topic,
            "graph_nodes": node_count,
            "graph_edges": edge_count,
            "top_influential": [
                {
                    "doi": p.doi,
                    "title": p.title,
                    "year": p.year,
                    "pagerank_score": p.pagerank_score,
                    "citation_count": p.citation_count,
                    "influence_tier": p.influence_tier,
                }
                for p in influential
            ],
            "bridge_papers": [
                {"doi": p.doi, "title": p.title, "betweenness": p.betweenness}
                for p in bridges
            ],
        }

    async def find_gaps(
        self,
        topic: str,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Find research gaps via semantic clustering."""
        papers = self.memory_manager.search_papers(topic, limit=500)
        if len(papers) < 10:
            logger.warning("Too few papers (%d) for gap detection on topic: %s", len(papers), topic)
            return {
                "topic": topic,
                "n_clusters": 0,
                "gaps_found": 0,
                "gaps": [],
                "cluster_summary": "Insufficient papers for clustering. Crawl more papers first.",
                "warning": "Fewer than 10 papers available",
            }

        finder = self._get_gap_finder()
        gap_config = self._config.get("gap_detection", {})
        density_threshold = gap_config.get("density_threshold", 0.45)

        gap_report = await finder.find_gaps(
            query=topic,
            n_clusters=n_clusters,
            density_threshold=density_threshold,
        )

        if _PROMETHEUS_AVAILABLE:
            gaps_found_total.inc(len(gap_report.gaps))

        return {
            "topic": topic,
            "n_clusters": gap_report.n_clusters,
            "gaps_found": len(gap_report.gaps),
            "gaps": [
                {
                    "cluster_id": g.cluster_id,
                    "cluster_size": g.cluster_size,
                    "density_score": g.density_score,
                    "centroid_keywords": g.centroid_keywords,
                    "llm_explanation": g.llm_explanation,
                    "suggested_directions": g.suggested_directions,
                    "urgency": g.urgency,
                }
                for g in gap_report.gaps
            ],
            "cluster_summary": finder.visualize_clusters_text(
                gap_report.clusters, gap_report.gaps
            ),
        }

    async def synthesize_literature(
        self,
        query: str,
        max_papers: int = 15,
        style: str = "academic",
    ) -> Dict[str, Any]:
        """Generate a literature review for the given query."""
        # Check synthesis cache (24h TTL)
        cached = self.memory_manager.get_synthesis(query, style)
        if cached:
            logger.info("Returning cached synthesis for: %s", query)
            return cached

        engine = self._get_synthesis_engine()
        t0 = time.time()
        review = await engine.generate_literature_review(
            query=query, max_papers=max_papers, style=style
        )
        elapsed = time.time() - t0

        if _PROMETHEUS_AVAILABLE:
            syntheses_generated_total.inc()
            llm_call_latency_seconds.labels(provider="synthesis", task="synthesize").observe(elapsed)

        result = {
            "query": review.query,
            "generated_at": review.generated_at,
            "style": review.style,
            "introduction": review.introduction,
            "key_themes": review.key_themes,
            "methodology_landscape": review.methodology_landscape,
            "state_of_the_art": review.state_of_the_art,
            "identified_gaps": review.identified_gaps,
            "future_directions": review.future_directions,
            "conclusion": review.conclusion,
            "references": [
                {
                    "index": ref.index,
                    "title": ref.title,
                    "authors": ref.authors,
                    "year": ref.year,
                    "url": ref.url,
                }
                for ref in review.references
            ],
            "quality_score": review.quality_score,
            "paper_count": review.paper_count,
            "llm_provider_used": getattr(review, "llm_provider_used", "unknown"),
        }

        self.memory_manager.save_synthesis(query, result, style)
        return result

    async def get_embeddings(self, texts: List[str], model_key: str = "bge-large") -> Dict[str, Any]:
        """Generate embeddings for a list of texts."""
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager.instance()
        model_name = HFModelManager.MODEL_REGISTRY.get(model_key, model_key)
        embeddings = mgr.encode(texts, model_key=model_key)
        return {
            "model": model_name,
            "dimensions": embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings[0]),
            "count": len(texts),
            "embeddings": embeddings.tolist(),
        }

    async def get_batch_embeddings(self, dois: List[str], model_key: str = "bge-large") -> List[Dict[str, Any]]:
        """Generate embeddings for papers identified by DOI."""
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager.instance()
        model_name = HFModelManager.MODEL_REGISTRY.get(model_key, model_key)
        results = []
        for doi in dois:
            paper_dict = self.memory_manager.get_paper(doi)
            if paper_dict and paper_dict.get("abstract"):
                text = f"{paper_dict['title']}. {paper_dict['abstract'][:512]}"
            elif paper_dict and paper_dict.get("title"):
                text = paper_dict["title"]
            else:
                text = doi
            emb = mgr.encode([text], model_key=model_key)
            results.append({
                "doi": doi,
                "embedding": emb[0].tolist() if emb.ndim > 1 else emb.tolist()[0],
                "model": model_name,
            })
        return results

    async def update_knowledge_base(self) -> Dict[str, Any]:
        """Trigger knowledge_updater to append new papers to SECOND-KNOWLEDGE-BRAIN.md."""
        from tools.knowledge_updater import KnowledgeUpdater
        updater = KnowledgeUpdater(memory_manager=self.memory_manager)
        return await updater.run_daily_update()

    def get_cost_report(self) -> Dict[str, Any]:
        """Return LLM cost summary from MemoryManager."""
        return self.memory_manager.get_cost_summary()

    def get_status(self) -> Dict[str, Any]:
        """Return current database statistics."""
        mm = self.memory_manager
        return {
            "papers_total": mm.count_papers(),
            "citations_total": mm.count_citations(),
            "clusters_total": mm.count_clusters(),
            "syntheses_cached": mm.count_syntheses(),
            "knowledge_hashes": mm.count_knowledge_hashes(),
            "agent_version": self.VERSION,
        }

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def start_scheduler(self):
        """Start APScheduler with daily self-update cron at 06:00."""
        if not _SCHEDULER_AVAILABLE:
            logger.warning("APScheduler not available; daily cron not started")
            return

        if self._scheduler is not None:
            logger.info("Scheduler already running")
            return

        self._scheduler = AsyncIOScheduler()
        self._scheduler.add_job(
            self._daily_self_update_loop,
            trigger=CronTrigger(hour=6, minute=0),
            id="daily_self_update",
            replace_existing=True,
            max_instances=1,
        )
        self._scheduler.start()
        logger.info("Daily self-update cron scheduled at 06:00")

    def stop_scheduler(self):
        if self._scheduler:
            self._scheduler.shutdown()
            self._scheduler = None

    async def _daily_self_update_loop(self):
        """Cron-triggered daily self-update: crawl new papers and update KB."""
        logger.info("Daily self-update loop starting")
        try:
            result = await self.update_knowledge_base()
            logger.info(
                "Daily self-update complete: %d papers added",
                result.get("papers_added", 0),
            )
        except Exception:
            logger.exception("Daily self-update loop failed")

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_pipeline_report(self, result: Dict[str, Any]) -> str:
        """Return a Markdown-formatted pipeline report."""
        lines = [
            f"# Research Pipeline Report",
            f"**Query:** {result.get('query', '')}",
            f"**Completed:** {result.get('pipeline_completed_at', '')}",
            f"**Elapsed:** {result.get('elapsed_seconds', 0):.1f}s",
            "",
            "## Crawl Results",
            f"- Papers found: {result['crawl'].get('papers_found', 0)}",
            f"- New papers: {result['crawl'].get('papers_new', 0)}",
            "",
            "## Citation Analysis",
            f"- Graph nodes: {result['citation_analysis'].get('graph_nodes', 0)}",
            f"- Graph edges: {result['citation_analysis'].get('graph_edges', 0)}",
            "",
            "## Research Gaps",
            f"- Gaps found: {result['gaps'].get('gaps_found', 0)}",
            "",
            "## Literature Synthesis",
            f"- Quality score: {result['synthesis'].get('quality_score', 0):.2f}",
            f"- Papers used: {result['synthesis'].get('paper_count', 0)}",
            "",
            "## Knowledge Base Update",
            f"- Papers added: {result['knowledge_base'].get('papers_added', 0)}",
        ]
        return "\n".join(lines)
