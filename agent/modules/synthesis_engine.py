"""
academic-research-enhanced — SynthesisEngine
Multi-paper literature review synthesis using Claude API.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional, TYPE_CHECKING

from agent.memory.memory_manager import Paper

if TYPE_CHECKING:
    from agent.memory.memory_manager import ResearchMemoryManager

logger = logging.getLogger("academic_agent.synthesis_engine")

SYNTHESIS_STYLES = ["academic", "technical", "survey", "executive"]


@dataclass
class Reference:
    index: int
    doi: str
    title: str
    authors: str
    year: int
    url: str


@dataclass
class LiteratureReview:
    query: str
    generated_at: str
    style: str
    introduction: str
    key_themes: List[str]
    methodology_landscape: str
    state_of_the_art: str
    identified_gaps: List[str]
    future_directions: str
    conclusion: str
    references: List[Reference]
    quality_score: float
    paper_count: int
    llm_provider_used: str = "unknown"


SYNTHESIS_PROMPT_TEMPLATE = """You are a world-class academic researcher synthesizing literature for: {query}

Papers provided (cite as [N]):
{context}

Write a comprehensive literature review with exactly these sections (use the exact headers):

## Introduction
Background and motivation for this research area. Why does this matter? (2-3 paragraphs)

## Key Themes
Identify 2-4 major research threads found across the papers. List each as a sub-bullet with citations.

## Methodology Landscape
Describe the main approaches, methods, and techniques used across these papers. Be specific.

## State of the Art
What are the current best results, benchmarks, and leading models or systems? Include specific numbers and year ranges.

## Identified Gaps
What is missing from the current literature? What problems remain unsolved?

## Future Directions
What specific experiments, datasets, or approaches should be pursued next?

## Conclusion
Synthesize the field direction in 1 cohesive paragraph.

Requirements:
- Use inline citations [1], [2], etc. matching the paper numbers above.
- Style: {style}
- Minimum 500 words total.
- Do NOT write [CITATION_NEEDED] — only cite papers in the provided list.
- Be specific: include year ranges, model names, metric values where available.
"""


class SynthesisEngine:
    """Synthesizes literature reviews from a set of papers using LLM + HuggingFace models."""

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    # ------------------------------------------------------------------
    # Paper selection
    # ------------------------------------------------------------------

    def select_papers(self, query: str, max_papers: int = 15) -> List[Paper]:
        """Select top papers for synthesis using BGE-large + reranker."""
        if not self.memory_manager:
            return []

        candidates = self.memory_manager.search_papers(query, limit=max_papers * 3)
        if not candidates:
            return []

        if len(candidates) <= max_papers:
            return candidates

        try:
            from tools.hf_model_manager import HFModelManager
            import numpy as np
            mgr = HFModelManager.instance()

            paper_texts = [f"{p.title}. {p.abstract[:512]}" for p in candidates]
            query_emb = mgr.encode([query], model_key="bge-large")[0]
            paper_embs = mgr.encode(paper_texts, model_key="bge-large")

            norms = np.linalg.norm(paper_embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            paper_embs_norm = paper_embs / norms
            query_norm = query_emb / (np.linalg.norm(query_emb) or 1.0)
            scores = paper_embs_norm @ query_norm

            top_indices = scores.argsort()[::-1][: max_papers * 2]
            top_candidates = [candidates[i] for i in top_indices]

            rerank_passages = [f"{p.title}. {p.abstract[:256]}" for p in top_candidates]
            rerank_scores = mgr.rerank(query, rerank_passages)

            if rerank_scores:
                reranked = sorted(
                    zip(top_candidates, rerank_scores),
                    key=lambda x: x[1],
                    reverse=True,
                )
                return [p for p, _ in reranked[:max_papers]]
            else:
                return top_candidates[:max_papers]

        except Exception as e:
            logger.warning("Semantic paper selection failed: %s; using score-based fallback", e)
            candidates.sort(key=lambda p: p.score, reverse=True)
            return candidates[:max_papers]

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def summarize_papers(self, papers: List[Paper]) -> List[str]:
        """Summarize each paper abstract to a 3-sentence digest using BART-CNN."""
        try:
            from tools.hf_model_manager import HFModelManager
            mgr = HFModelManager.instance()
            summaries = []
            for p in papers:
                if len(p.abstract) > 100:
                    summary = mgr.summarize(p.abstract, max_length=150)
                else:
                    summary = p.abstract or p.title
                summaries.append(summary)
            return summaries
        except Exception as e:
            logger.warning("BART summarization failed: %s; using truncated abstracts", e)
            return [p.abstract[:300] if p.abstract else p.title for p in papers]

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_synthesis_context(self, papers: List[Paper], summaries: List[str]) -> str:
        """Build the numbered context block for the LLM prompt."""
        lines = []
        for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " et al."
            lines.append(
                f"[{i}] {paper.title}\n"
                f"    Authors: {authors_str} | Year: {paper.year} | Source: {paper.source}\n"
                f"    Summary: {summary}\n"
                f"    URL: {paper.url}"
            )
        return "\n\n".join(lines)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self,
        query: str,
        papers: List[Paper],
        summaries: List[str],
        style: str = "academic",
    ) -> LiteratureReview:
        """Call LLM to generate the literature review."""
        if style not in SYNTHESIS_STYLES:
            style = "academic"

        context = self.build_synthesis_context(papers, summaries)
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(query=query, context=context, style=style)

        raw_text = ""
        provider_used = "unknown"

        try:
            from tools.llm_client import UnifiedLLMClient
            client = UnifiedLLMClient()
            raw_text = await client.complete(
                prompt=prompt,
                task="literature_synthesis",
                max_tokens=4000,
                temperature=0.3,
            )
            provider_used = getattr(client, "_last_provider", "unknown")
        except Exception as e:
            logger.error("LLM synthesis failed: %s", e)
            raw_text = self._fallback_synthesis(query, papers, summaries)
            provider_used = "fallback"

        review = self._parse_review(raw_text, query, style, papers, provider_used)
        review.quality_score = self._compute_quality_score(review)
        return review

    def _fallback_synthesis(
        self, query: str, papers: List[Paper], summaries: List[str]
    ) -> str:
        """Minimal fallback when LLM is unavailable."""
        lines = [
            f"## Introduction\nThis review covers {len(papers)} papers on: {query}.",
            "",
            "## Key Themes\n" + "\n".join(f"- {p.title} [{i+1}]" for i, p in enumerate(papers[:5])),
            "",
            "## Methodology Landscape\nVarious methodological approaches are represented in these papers.",
            "",
            "## State of the Art\nSee individual paper summaries below.",
            "",
            "## Identified Gaps\n- Further research is needed.",
            "",
            "## Future Directions\n- Continued investigation of this topic area is recommended.",
            "",
            "## Conclusion\n" + f"This collection of {len(papers)} papers advances the field of {query}.",
        ]
        return "\n".join(lines)

    def _parse_review(
        self,
        raw_text: str,
        query: str,
        style: str,
        papers: List[Paper],
        provider_used: str,
    ) -> LiteratureReview:
        """Parse LLM output into structured LiteratureReview."""

        def _extract_section(text: str, header: str, next_headers: List[str]) -> str:
            pattern = rf"##\s*{re.escape(header)}\s*\n(.*?)(?=##\s*(?:{'|'.join(re.escape(h) for h in next_headers)})|$)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return ""

        all_headers = [
            "Introduction", "Key Themes", "Methodology Landscape",
            "State of the Art", "Identified Gaps", "Future Directions", "Conclusion"
        ]

        introduction = _extract_section(raw_text, "Introduction", all_headers[1:])
        key_themes_raw = _extract_section(raw_text, "Key Themes", all_headers[2:])
        methodology = _extract_section(raw_text, "Methodology Landscape", all_headers[3:])
        sota = _extract_section(raw_text, "State of the Art", all_headers[4:])
        gaps_raw = _extract_section(raw_text, "Identified Gaps", all_headers[5:])
        future = _extract_section(raw_text, "Future Directions", all_headers[6:])
        conclusion = _extract_section(raw_text, "Conclusion", [])

        if not introduction and not conclusion:
            introduction = raw_text[:1000]
            conclusion = raw_text[-500:] if len(raw_text) > 500 else raw_text

        key_themes = [
            line.lstrip("-\u2022 ").strip()
            for line in key_themes_raw.split("\n")
            if line.strip().startswith(("-", "\u2022", "*")) and len(line.strip()) > 5
        ] or ([key_themes_raw[:200]] if key_themes_raw else ["See full review text"])

        identified_gaps = [
            line.lstrip("-\u2022 ").strip()
            for line in gaps_raw.split("\n")
            if line.strip().startswith(("-", "\u2022", "*")) and len(line.strip()) > 5
        ] or ([gaps_raw[:200]] if gaps_raw else ["Not specified"])

        references = []
        for i, p in enumerate(papers, 1):
            authors_str = ", ".join(p.authors[:3])
            if len(p.authors) > 3:
                authors_str += " et al."
            references.append(Reference(
                index=i,
                doi=p.doi,
                title=p.title,
                authors=authors_str,
                year=p.year,
                url=p.url,
            ))

        return LiteratureReview(
            query=query,
            generated_at=datetime.now(timezone.utc).isoformat(),
            style=style,
            introduction=introduction,
            key_themes=key_themes[:10],
            methodology_landscape=methodology,
            state_of_the_art=sota,
            identified_gaps=identified_gaps[:10],
            future_directions=future,
            conclusion=conclusion,
            references=references,
            quality_score=0.0,
            paper_count=len(papers),
            llm_provider_used=provider_used,
        )

    def _compute_quality_score(self, review: LiteratureReview) -> float:
        """Compute quality score (0..1) from completeness and citation checks."""
        score = 0.0
        max_score = 7.0

        full_text = " ".join([
            review.introduction,
            " ".join(review.key_themes),
            review.methodology_landscape,
            review.state_of_the_art,
            " ".join(review.identified_gaps),
            review.future_directions,
            review.conclusion,
        ])

        word_count = len(full_text.split())
        if word_count >= 500:
            score += 2.0
        elif word_count >= 200:
            score += 1.0

        citation_matches = re.findall(r"\[\d+\]", full_text)
        unique_citations = set(citation_matches)
        if len(unique_citations) >= 5:
            score += 2.0
        elif len(unique_citations) >= 3:
            score += 1.0

        placeholders = ["[CITATION_NEEDED]", "[TODO]", "[PLACEHOLDER]"]
        if not any(ph in full_text for ph in placeholders):
            score += 1.0

        sections = [
            review.introduction, review.methodology_landscape,
            review.state_of_the_art, review.future_directions, review.conclusion
        ]
        filled_sections = sum(1 for s in sections if s and len(s) > 50)
        score += filled_sections / len(sections)

        if len(review.references) >= 3:
            score += 1.0

        return min(1.0, score / max_score)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def generate_literature_review(
        self,
        query: str,
        max_papers: int = 15,
        style: str = "academic",
    ) -> LiteratureReview:
        """Full pipeline: select -> summarize -> synthesize."""
        papers = self.select_papers(query, max_papers)

        if len(papers) < 3:
            logger.warning("Only %d papers found for synthesis; quality may be low", len(papers))

        summaries = self.summarize_papers(papers)
        review = await self.synthesize(query, papers, summaries, style)

        return review
