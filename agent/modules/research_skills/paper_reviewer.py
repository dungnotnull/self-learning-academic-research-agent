"""
academic-research-enhanced — PaperReviewer

Runs a pre-submission review across five dimensions: macro logic,
writing details, English grammar, LaTeX formatting, and figure quality.
Uses the Supervisor-Skills pre-submission-reviewer methodology.

Based on: https://github.com/HKUSTDial/Supervisor-Skills
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.memory.memory_manager import ResearchMemoryManager

logger = logging.getLogger("academic_agent.paper_reviewer")


PAPER_REVIEWER_SYSTEM_PROMPT = """You are a senior reviewer at a top venue (NeurIPS, ICML, ICLR, VLDB, SIGMOD, KDD) performing a pre-submission review of a research paper. You evaluate across five dimensions:

DIMENSION 1 — Macro Logic:
- Introduction flowchart intact (Background → Limitations → Goal → Challenges → Methodology → Contributions)
- Contributions map one-to-one with methodology modules and section numbers
- Experiments validate main claims
- Running example consistent across Introduction, Methodology, Experiments

DIMENSION 2 — Writing Details:
- Every paragraph has a topic sentence
- No orphan paragraphs
- Paragraphs not over 10 lines
- No repeated or redundant passages
- Abstract covers problem, method, result

DIMENSION 3 — English Grammar (non-native author focus):
- Article use (a, an, the)
- Subject-verb agreement
- Tense consistency
- Passive-voice overuse
- Which vs that
- Sentence length
- Chinglish patterns

DIMENSION 4 — LaTeX Format:
- Equation numbering contiguous and referenced
- Figures and tables have captions
- Citations use non-breaking tilde (ResNet~\\\\cite{X})
- Labels use underscores
- Vector figure format
- Page-limit compliance

DIMENSION 5 — Figure Quality:
- Vector format
- Font size large enough
- Colour-blind-safe palette
- Self-contained caption
- No chartjunk

SEVERITY TAXONOMY:
- CRITICAL: blocks submission (broken flowchart, missing baseline, raster figure)
- MAJOR: reviewers will flag (3+ topic-sentence issues, banned words)
- MINOR: polish (two long sentences, default styling)

BANNED WORDS (AI-tone): innovative, pioneering, revolutionary paradigm, transformative framework, superior, surpass, excel, remarkable, unprecedented, breakthrough performance, general-purpose, is capable of, notably, yet (as conjunction), yielding, at its essence, encompass, differentiate, reveal, underscore, pave the way for, highlight the potential of, profound challenges, stems from, rigid, impede.

EM-DASHES: Em-dashes (—) used as sentence connectors are banned. Use parentheses or commas instead.

Output a structured review with findings per dimension, severity tags, and a final score.
"""

PAPER_REVIEWER_TEMPLATE = """Review the following paper draft:

RESEARCH AREA: {research_area}
PAPER TITLE: {paper_title}
TARGET VENUE: {target_venue}
PAPER TEXT:
{paper_text}

Perform the complete five-dimension review as specified in your instructions. Quote specific text for each finding.
"""


@dataclass
class PaperReview:
    """Structured result of a pre-submission review."""
    research_area: str
    paper_title: str
    target_venue: str
    summary: Dict[str, int]  # {"CRITICAL": n, "MAJOR": m, "MINOR": k}
    dimensions: Dict[str, List[Dict[str, str]]]  # {"macro_logic": [{finding, severity, fix}], ...}
    banned_vocabulary_findings: List[Dict[str, str]]
    em_dash_findings: List[Dict[str, str]]
    final_score: float  # 1-10
    submission_recommendation: str  # "Ready to submit" | "Needs 1-2 days" | "Needs major revision"
    top_three_fixes: List[str]
    raw_review: str
    reviewed_at: str
    llm_provider_used: str = "unknown"


class PaperReviewer:
    """Pre-submission paper review using the Supervisor-Skills methodology."""

    BANNED_WORDS = [
        "innovative", "pioneering", "revolutionary paradigm", "transformative framework",
        "superior", "surpass", "excel", "remarkable", "unprecedented",
        "breakthrough performance", "general-purpose", "is capable of", "notably",
        "yielding", "at its essence", "encompass", "differentiate", "reveal",
        "underscore", "pave the way for", "highlight the potential of",
        "profound challenges", "stems from", "rigid", "impede",
    ]

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    async def review(
        self,
        research_area: str,
        paper_title: str = "",
        target_venue: str = "",
        paper_text: str = "",
        max_tokens: int = 6000,
    ) -> PaperReview:
        """Run a five-dimension pre-submission review."""
        from tools.llm_client import UnifiedLLMClient

        client = UnifiedLLMClient(memory_manager=self.memory_manager)
        prompt = PAPER_REVIEWER_TEMPLATE.format(
            research_area=research_area,
            paper_title=paper_title or "Untitled",
            target_venue=target_venue or "Top venue",
            paper_text=paper_text or "(No text provided — review based on available information)",
        )

        raw = await client.complete(
            prompt=prompt,
            system=PAPER_REVIEWER_SYSTEM_PROMPT,
            task="paper_review",
            max_tokens=max_tokens,
            temperature=0.2,
        )
        provider = getattr(client, "_last_provider", "unknown")

        # Also run banned vocabulary scan
        banned_findings = self._scan_banned_vocabulary(paper_text)
        em_dash_findings = self._scan_em_dashes(paper_text)

        return self._parse_review(raw, research_area, paper_title, target_venue, provider, banned_findings, em_dash_findings)

    def _scan_banned_vocabulary(self, text: str) -> List[Dict[str, str]]:
        """Scan for banned AI-tone vocabulary."""
        findings = []
        text_lower = text.lower()
        for word in self.BANNED_WORDS:
            if word.lower() in text_lower:
                # Find context
                idx = text_lower.find(word.lower())
                start = max(0, idx - 40)
                end = min(len(text), idx + len(word) + 40)
                context = text[start:end]
                findings.append({
                    "word": word,
                    "severity": "MAJOR" if text_lower.count(word.lower()) >= 3 else "MINOR",
                    "context": f"...{context}...",
                })
        return findings

    def _scan_em_dashes(self, text: str) -> List[Dict[str, str]]:
        """Scan for em-dashes used as sentence connectors."""
        findings = []
        em_dash_patterns = ["—", "–", "---"]
        for pattern in em_dash_patterns:
            count = text.count(pattern)
            if count > 0:
                findings.append({
                    "pattern": pattern,
                    "count": count,
                    "severity": "MAJOR" if count >= 5 else "MINOR",
                    "note": "Em-dashes as sentence connectors are banned; use parentheses or commas.",
                })
        return findings

    def _parse_review(
        self, raw: str, research_area: str, paper_title: str, target_venue: str,
        provider: str, banned_findings: List[Dict[str, str]], em_dash_findings: List[Dict[str, str]]
    ) -> PaperReview:
        """Parse LLM output into structured PaperReview."""
        # Extract severity counts
        critical = len(re.findall(r"CRITICAL", raw, re.IGNORECASE))
        major = len(re.findall(r"MAJOR", raw, re.IGNORECASE)) - critical
        minor = len(re.findall(r"MINOR", raw, re.IGNORECASE))

        # Extract findings per dimension
        dimensions = {
            "macro_logic": [],
            "writing_details": [],
            "english_grammar": [],
            "latex_format": [],
            "figure_quality": [],
        }
        dim_names = {
            "macro_logic": ["Dimension 1", "Macro Logic", "macro logic"],
            "writing_details": ["Dimension 2", "Writing Details", "writing detail"],
            "english_grammar": ["Dimension 3", "English Grammar", "grammar"],
            "latex_format": ["Dimension 4", "LaTeX Format", "latex format"],
            "figure_quality": ["Dimension 5", "Figure Quality", "figure quality"],
        }
        for dim_key, search_terms in dim_names.items():
            for term in search_terms:
                pattern = rf"{re.escape(term)}(.*?)(?=###\\s*(?:Dimension|Summary)|$)"
                match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
                if match:
                    section = match.group(1)
                    for line in section.split("\n"):
                        line = line.strip()
                        if line and ("CRITICAL" in line or "MAJOR" in line or "MINOR" in line):
                            severity = "MINOR"
                            if "CRITICAL" in line:
                                severity = "CRITICAL"
                            elif "MAJOR" in line:
                                severity = "MAJOR"
                            dimensions[dim_key].append({
                                "finding": line.lstrip("-|* "),
                                "severity": severity,
                                "fix": "",
                            })
                    break

        # Extract final score
        score = 5.0
        score_match = re.search(r"(?:Final score|Score)[^\\n]*?(\\d+(?:\\.\\d+)?)", raw, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
            except ValueError:
                pass

        # Extract recommendation
        recommendation = "Needs major revision before submission"
        for rec in ["Ready to submit", "Needs 1-2 days more work", "Needs major revision"]:
            if rec.lower() in raw.lower():
                recommendation = rec
                break

        # Top fixes
        fixes = []
        fix_section = re.search(r"(?:Top three|Top 3)[^\\n]*?fix[^\\n]*\\n(.*?)(?=###|$)", raw, re.DOTALL | re.IGNORECASE)
        if fix_section:
            for line in fix_section.group(1).split("\n"):
                line = line.strip()
                if line and re.match(r"^\\d", line):
                    fixes.append(re.sub(r"^\\d+[\\.)\\]\\s]*", "", line))
        if not fixes:
            fixes = ["Review the full output for prioritized fixes."]

        return PaperReview(
            research_area=research_area,
            paper_title=paper_title or "Untitled",
            target_venue=target_venue or "Top venue",
            summary={"CRITICAL": critical, "MAJOR": max(0, major), "MINOR": minor},
            dimensions=dimensions,
            banned_vocabulary_findings=banned_findings,
            em_dash_findings=em_dash_findings,
            final_score=min(10.0, max(1.0, score)),
            submission_recommendation=recommendation,
            top_three_fixes=fixes[:3],
            raw_review=raw,
            reviewed_at=datetime.now(timezone.utc).isoformat(),
            llm_provider_used=provider,
        )
