"""
academic-research-enhanced — ResearchOrchestrator (v2.0)
Core agent decision loop. Wires all modules together including research skills.
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

    VERSION = "2.0.0"

    def __init__(self, config_path: Optional[str] = None):
        from agent.memory.memory_manager import ResearchMemoryManager
        self.memory_manager = ResearchMemoryManager()
        self._config = self._load_config(config_path)
        self._crawler = None
        self._citation_analyzer = None
        self._gap_finder = None
        self._synthesis_engine = None
        self._scheduler = None
        # v2.0: Research skills
        self._idea_evaluator = None
        self._paper_planner = None
        self._intro_drafter = None
        self._figure_designer = None
        self._paper_reviewer = None
        self._research_workflow = None

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

    def _get_idea_evaluator(self):
        if self._idea_evaluator is None:
            from agent.modules.research_skills.idea_evaluator import IdeaEvaluator
            self._idea_evaluator = IdeaEvaluator(memory_manager=self.memory_manager)
        return self._idea_evaluator

    def _get_paper_planner(self):
        if self._paper_planner is None:
            from agent.modules.research_skills.paper_planner import PaperPlanner
            self._paper_planner = PaperPlanner(memory_manager=self.memory_manager)
        return self._paper_planner

    def _get_intro_drafter(self):
        if self._intro_drafter is None:
            from agent.modules.research_skills.intro_drafter import IntroDrafter
            self._intro_drafter = IntroDrafter(memory_manager=self.memory_manager)
        return self._intro_drafter

    def _get_figure_designer(self):
        if self._figure_designer is None:
            from agent.modules.research_skills.figure_designer import FigureDesigner
            self._figure_designer = FigureDesigner(memory_manager=self.memory_manager)
        return self._figure_designer

    def _get_paper_reviewer(self):
        if self._paper_reviewer is None:
            from agent.modules.research_skills.paper_reviewer import PaperReviewer
            self._paper_reviewer = PaperReviewer(memory_manager=self.memory_manager)
        return self._paper_reviewer

    def _get_research_workflow(self):
        if self._research_workflow is None:
            from agent.modules.research_skills.research_workflow import ResearchWorkflow
            self._research_workflow = ResearchWorkflow(memory_manager=self.memory_manager)
        return self._research_workflow

    # ------------------------------------------------------------------
    # Core async pipeline methods (v1.0 — unchanged)
    # ------------------------------------------------------------------

    async def run_full_pipeline(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        max_papers: int = 50,
    ) -> Dict[str, Any]:
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]

        logger.info("Starting full pipeline for query: %s", query)
        pipeline_start = time.time()

        crawl_result = await self.crawl_papers(
            query=query, sources=sources, max_results=max_papers, days_back=7
        )
        analyze_result = await self.analyze_citations(topic=query, min_citations=3)
        gap_result = await self.find_gaps(topic=query)
        synthesis_result = await self.synthesize_literature(
            query=query, max_papers=min(max_papers, 15), style="academic"
        )
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

    async def crawl_papers(self, query: str, sources: Optional[List[str]] = None,
                           max_results: int = 50, days_back: int = 7) -> Dict[str, Any]:
        if sources is None:
            sources = ["arxiv", "semantic_scholar"]
        crawler = self._get_crawler()
        papers = await crawler.crawl(query=query, sources=sources, max_results=max_results, days_back=days_back)
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

    async def analyze_citations(self, topic: str, min_citations: int = 5) -> Dict[str, Any]:
        papers = self.memory_manager.search_papers(topic, limit=200)
        if len(papers) < 5:
            logger.warning("Too few papers (%d) for citation analysis of topic: %s", len(papers), topic)
            return {"topic": topic, "graph_nodes": 0, "graph_edges": 0, "top_influential": [], "bridge_papers": [], "warning": "Fewer than 5 papers available; crawl more papers first"}
        analyzer = self._get_citation_analyzer()
        graph = await analyzer.build_citation_graph(papers)
        influential = analyzer.identify_influential(graph, top_k=10)
        bridges = analyzer.identify_bridge_papers(graph, top_k=5)
        return {
            "topic": topic,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "top_influential": [
                {"doi": p.doi, "title": p.title, "year": p.year, "pagerank_score": p.pagerank_score, "citation_count": p.citation_count, "influence_tier": p.influence_tier}
                for p in influential
            ],
            "bridge_papers": [{"doi": p.doi, "title": p.title, "betweenness": p.betweenness} for p in bridges],
        }

    async def find_gaps(self, topic: str, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        papers = self.memory_manager.search_papers(topic, limit=500)
        if len(papers) < 10:
            logger.warning("Too few papers (%d) for gap detection on topic: %s", len(papers), topic)
            return {"topic": topic, "n_clusters": 0, "gaps_found": 0, "gaps": [], "cluster_summary": "Insufficient papers for clustering. Crawl more papers first.", "warning": "Fewer than 10 papers available"}
        finder = self._get_gap_finder()
        gap_config = self._config.get("gap_detection", {})
        density_threshold = gap_config.get("density_threshold", 0.45)
        gap_report = await finder.find_gaps(query=topic, n_clusters=n_clusters, density_threshold=density_threshold)
        if _PROMETHEUS_AVAILABLE:
            gaps_found_total.inc(len(gap_report.gaps))
        return {
            "topic": topic,
            "n_clusters": gap_report.n_clusters,
            "gaps_found": len(gap_report.gaps),
            "gaps": [
                {"cluster_id": g.cluster_id, "cluster_size": g.cluster_size, "density_score": g.density_score,
                 "centroid_keywords": g.centroid_keywords, "llm_explanation": g.llm_explanation,
                 "suggested_directions": g.suggested_directions, "urgency": g.urgency}
                for g in gap_report.gaps
            ],
            "cluster_summary": finder.visualize_clusters_text(gap_report.clusters, gap_report.gaps),
        }

    async def synthesize_literature(self, query: str, max_papers: int = 15, style: str = "academic") -> Dict[str, Any]:
        cached = self.memory_manager.get_synthesis(query, style)
        if cached:
            logger.info("Returning cached synthesis for: %s", query)
            return cached
        engine = self._get_synthesis_engine()
        t0 = time.time()
        review = await engine.generate_literature_review(query=query, max_papers=max_papers, style=style)
        elapsed = time.time() - t0
        if _PROMETHEUS_AVAILABLE:
            syntheses_generated_total.inc()
            llm_call_latency_seconds.labels(provider="synthesis", task="synthesize").observe(elapsed)
        result = {
            "query": review.query, "generated_at": review.generated_at, "style": review.style,
            "introduction": review.introduction, "key_themes": review.key_themes,
            "methodology_landscape": review.methodology_landscape, "state_of_the_art": review.state_of_the_art,
            "identified_gaps": review.identified_gaps, "future_directions": review.future_directions,
            "conclusion": review.conclusion,
            "references": [{"index": ref.index, "title": ref.title, "authors": ref.authors, "year": ref.year, "url": ref.url} for ref in review.references],
            "quality_score": review.quality_score, "paper_count": review.paper_count,
            "llm_provider_used": getattr(review, "llm_provider_used", "unknown"),
        }
        self.memory_manager.save_synthesis(query, result, style)
        return result

    async def get_embeddings(self, texts: List[str], model_key: str = "bge-large") -> Dict[str, Any]:
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager.instance()
        model_name = HFModelManager.MODEL_REGISTRY.get(model_key, model_key)
        embeddings = mgr.encode(texts, model_key=model_key)
        return {"model": model_name, "dimensions": embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings[0]), "count": len(texts), "embeddings": embeddings.tolist()}

    async def get_batch_embeddings(self, dois: List[str], model_key: str = "bge-large") -> List[Dict[str, Any]]:
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
            results.append({"doi": doi, "embedding": emb[0].tolist() if emb.ndim > 1 else emb.tolist()[0], "model": model_name})
        return results

    async def update_knowledge_base(self) -> Dict[str, Any]:
        from tools.knowledge_updater import KnowledgeUpdater
        updater = KnowledgeUpdater(memory_manager=self.memory_manager)
        return await updater.run_daily_update()

    def get_cost_report(self) -> Dict[str, Any]:
        return self.memory_manager.get_cost_summary()

    def get_status(self) -> Dict[str, Any]:
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
    # v2.0: Research skill pipeline methods
    # ------------------------------------------------------------------

    async def evaluate_idea(self, **kwargs) -> Dict[str, Any]:
        """Evaluate a research idea using the five-dimension framework."""
        evaluator = self._get_idea_evaluator()
        result = await evaluator.evaluate(**kwargs)
        return {
            "idea_summary": result.idea_summary,
            "paper_type": result.paper_type,
            "one_sentence_story": result.one_sentence_story,
            "fatal_flaws": result.fatal_flaws,
            "has_critical_flaw": result.has_critical_flaw,
            "lifecycle_category": result.lifecycle_category,
            "capability_match": result.capability_match,
            "dimension_scores": result.dimension_scores,
            "disruptive_potential": result.disruptive_potential,
            "feasibility_risks": result.feasibility_risks,
            "verdict": result.verdict,
            "top_three_actions": result.top_three_actions,
            "evaluated_at": result.evaluated_at,
            "llm_provider_used": result.llm_provider_used,
        }

    async def plan_technique_paper(self, **kwargs) -> Dict[str, Any]:
        """Plan a technique paper logical skeleton."""
        planner = self._get_paper_planner()
        result = await planner.plan_technique_paper(**kwargs)
        return {
            "research_area": result.research_area,
            "paper_type": result.paper_type,
            "positioning": result.positioning,
            "thinking_template": result.thinking_template,
            "consistency_checks": result.consistency_checks,
            "gaps": result.gaps,
            "section_skeleton": result.section_skeleton,
            "next_skill": result.next_skill,
            "planned_at": result.planned_at,
            "llm_provider_used": result.llm_provider_used,
        }

    async def plan_benchmark_paper(self, **kwargs) -> Dict[str, Any]:
        """Plan a benchmark paper logical skeleton."""
        planner = self._get_paper_planner()
        result = await planner.plan_benchmark_paper(**kwargs)
        return {
            "research_area": result.research_area,
            "paper_type": result.paper_type,
            "positioning": result.positioning,
            "thinking_template": result.thinking_template,
            "consistency_checks": result.consistency_checks,
            "gaps": result.gaps,
            "section_skeleton": result.section_skeleton,
            "next_skill": result.next_skill,
            "planned_at": result.planned_at,
            "llm_provider_used": result.llm_provider_used,
        }

    async def draft_introduction(self, **kwargs) -> Dict[str, Any]:
        """Draft a six-paragraph Introduction outline."""
        drafter = self._get_intro_drafter()
        result = await drafter.draft(**kwargs)
        return {
            "research_area": result.research_area,
            "paper_type": result.paper_type,
            "type_positioning_rationale": result.type_positioning_rationale,
            "paragraphs": result.paragraphs,
            "challenge_module_mapping": result.challenge_module_mapping,
            "contribution_section_mapping": result.contribution_section_mapping,
            "consistency_report": result.consistency_report,
            "severity_summary": result.severity_summary,
            "top_three_actions": result.top_three_actions,
            "drafted_at": result.drafted_at,
            "llm_provider_used": result.llm_provider_used,
        }

    async def design_figure(self, **kwargs) -> Dict[str, Any]:
        """Get figure design advice for a paper figure."""
        designer = self._get_figure_designer()
        result = await designer.design(**kwargs)
        return {
            "research_area": result.research_area,
            "figure_type": result.figure_type,
            "paradigm_recommendation": result.paradigm_recommendation,
            "paradigm_rationale": result.paradigm_rationale,
            "layout_sketch": result.layout_sketch,
            "labelling_guidance": result.labelling_guidance,
            "tool_suggestion_primary": result.tool_suggestion_primary,
            "tool_suggestion_alternative": result.tool_suggestion_alternative,
            "tool_rationale": result.tool_rationale,
            "audit_results": result.audit_results,
            "severity_summary": result.severity_summary,
            "top_three_actions": result.top_three_actions,
            "designed_at": result.designed_at,
            "llm_provider_used": result.llm_provider_used,
        }

    async def review_paper(self, **kwargs) -> Dict[str, Any]:
        """Run a five-dimension pre-submission review."""
        reviewer = self._get_paper_reviewer()
        result = await reviewer.review(**kwargs)
        return {
            "research_area": result.research_area,
            "paper_title": result.paper_title,
            "target_venue": result.target_venue,
            "summary": result.summary,
            "dimensions": result.dimensions,
            "banned_vocabulary_findings": result.banned_vocabulary_findings,
            "em_dash_findings": result.em_dash_findings,
            "final_score": result.final_score,
            "submission_recommendation": result.submission_recommendation,
            "top_three_fixes": result.top_three_fixes,
            "reviewed_at": result.reviewed_at,
            "llm_provider_used": result.llm_provider_used,
        }

    async def plan_research_workflow(self, **kwargs) -> Dict[str, Any]:
        """Plan an AI-assisted research workflow."""
        workflow = self._get_research_workflow()
        result = await workflow.plan(**kwargs)
        return {
            "research_area": result.research_area,
            "primary_phase": result.primary_phase,
            "secondary_phases": result.secondary_phases,
            "behavioural_rules_acknowledged": result.behavioural_rules_acknowledged,
            "workflow_table": result.workflow_table,
            "tool_recommendations": result.tool_recommendations,
            "red_line_reminders": result.red_line_reminders,
            "verification_points": result.verification_points,
            "ai_disclosure_requirements": result.ai_disclosure_requirements,
            "planned_at": result.planned_at,
            "llm_provider_used": result.llm_provider_used,
        }

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def start_scheduler(self):
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
        logger.info("Daily self-update loop starting")
        try:
            result = await self.update_knowledge_base()
            logger.info("Daily self-update complete: %d papers added", result.get("papers_added", 0))
        except Exception:
            logger.exception("Daily self-update loop failed")
