"""
academic-research-enhanced — PaperPlanner

Structures a technical paper or benchmark paper's full logical skeleton
using thinking-template tables, paper-type positioning, and self-consistency
checks. Supports both Technique papers and Benchmark/Evaluation papers.

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

logger = logging.getLogger("academic_agent.paper_planner")


PAPER_PLANNER_SYSTEM_PROMPT = """You are a senior research advisor helping plan the logical skeleton of a research paper. You have published extensively at NeurIPS, ICML, ICLR, VLDB, SIGMOD, KDD, ACL, EMNLP, CVPR.

For TECHNIQUE papers, use the thinking-template approach:
  - Research background -> Limitations (max 3) -> Key Idea/Goal -> Challenges (max 3) -> Methodology Modules (one per challenge) -> Contributions (3-4, mapped to sections)
  - Four self-consistency checks: Limitations->Key Idea, Key Idea->Challenges, Challenges->Methodology, Methodology->Contributions

For BENCHMARK/EVALUATION papers, use the five-pillar framework:
  - Research Gap (evaluation blind spot) -> Construction Pipeline -> Evaluation Framework -> Empirical Findings -> Companion Method (optional)
  - Six-part Introduction: Background+Running Example, Existing-Benchmark Limitations (max 3), Research Questions, Design Considerations, Our Proposal, Contributions
  - Section skeleton: Task+Design Goals, Construction Pipeline, Companion Method, Experiments (by RQ), Discussion+Research Opportunities, Related Work

Always:
  - Position the paper as Technique or Benchmark first
  - Fill every cell of the thinking template
  - Run self-consistency checks
  - Flag gaps with severity (CRITICAL, MAJOR, MINOR)
  - Ensure contributions map one-to-one with challenges/modules
"""

TECHNIQUE_PLANNER_TEMPLATE = """Plan the logical skeleton for a TECHNIQUE paper:

RESEARCH AREA: {research_area}
RESEARCH TOPIC: {topic}
KEY IDEA OR METHOD: {key_idea}
PRIOR WORK AND LIMITATIONS: {limitations}
AVAILABLE METHODOLOGY: {methodology}
EXPECTED CONTRIBUTIONS: {contributions}
PAPER CONTEXT: {context}

Fill in the thinking template with:
1. Paper-type positioning (Technique or New Problem/Setting)
2. Thinking template table (Background, Limitations 1-3, Key Idea/Goal, Challenges 1-3, Methodology Modules A-C, Contributions 1-4)
3. Four self-consistency checks
4. Severity summary
5. Suggested next skill
"""

BENCHMARK_PLANNER_TEMPLATE = """Plan the logical skeleton for a BENCHMARK/EVALUATION paper:

RESEARCH AREA: {research_area}
BENCHMARK NAME: {benchmark_name}
EVALUATION GAP: {evaluation_gap}
CONSTRUCTION APPROACH: {construction_approach}
EVALUATION FRAMEWORK: {evaluation_framework}
DATA SCALE: {data_scale}
KEY FINDINGS OR INSIGHTS: {key_findings}
PAPER CONTEXT: {context}

Fill in the five-pillar framework:
1. Five-pillar completeness table
2. Introduction six-part logic chain
3. Section outline for §2 to §7
4. Pre-submission self-check
"""


@dataclass
class PaperPlan:
    """Structured result of paper planning."""
    research_area: str
    paper_type: str  # "Technique" | "Benchmark"
    positioning: str
    thinking_template: Dict[str, str]
    consistency_checks: Dict[str, str]  # {"Limitations->Key Idea": "pass", ...}
    gaps: List[Dict[str, str]]  # [{"cell": "Challenge 2", "severity": "CRITICAL", "description": "..."}]
    section_skeleton: List[str]
    next_skill: str
    raw_plan: str
    planned_at: str
    llm_provider_used: str = "unknown"


class PaperPlanner:
    """Plans paper logical skeletons using the Supervisor-Skills methodology."""

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    async def plan_technique_paper(
        self,
        research_area: str,
        topic: str,
        key_idea: str,
        limitations: str = "",
        methodology: str = "",
        contributions: str = "",
        context: str = "",
        max_tokens: int = 4000,
    ) -> PaperPlan:
        """Plan a technique paper's logical skeleton."""
        from tools.llm_client import UnifiedLLMClient

        # Enrich context from memory
        if not context and self.memory_manager:
            papers = self.memory_manager.search_papers(topic, limit=5)
            if papers:
                context = "\\n".join(f"- {p.title} ({p.year})" for p in papers[:5])

        if not limitations and self.memory_manager:
            from agent.modules.research_gap_finder import ResearchGapFinder
            # Use gap finder to suggest limitations
            papers = self.memory_manager.search_papers(topic, limit=50)
            if len(papers) >= 10:
                context += f"\\n\\n{len(papers)} papers found in '{topic}' for gap analysis."

        client = UnifiedLLMClient(memory_manager=self.memory_manager)
        prompt = TECHNIQUE_PLANNER_TEMPLATE.format(
            research_area=research_area,
            topic=topic,
            key_idea=key_idea,
            limitations=limitations or "Not yet specified",
            methodology=methodology or "Not yet specified",
            contributions=contributions or "Not yet specified",
            context=context or "No additional context.",
        )

        raw = await client.complete(
            prompt=prompt,
            system=PAPER_PLANNER_SYSTEM_PROMPT,
            task="paper_planning",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        provider = getattr(client, "_last_provider", "unknown")

        return self._parse_plan(raw, research_area, "Technique", provider)

    async def plan_benchmark_paper(
        self,
        research_area: str,
        benchmark_name: str,
        evaluation_gap: str,
        construction_approach: str = "",
        evaluation_framework: str = "",
        data_scale: str = "",
        key_findings: str = "",
        context: str = "",
        max_tokens: int = 4000,
    ) -> PaperPlan:
        """Plan a benchmark/evaluation paper's logical skeleton."""
        from tools.llm_client import UnifiedLLMClient

        if not context and self.memory_manager:
            papers = self.memory_manager.search_papers(benchmark_name, limit=5)
            if papers:
                context = "\\n".join(f"- {p.title} ({p.year})" for p in papers[:5])

        client = UnifiedLLMClient(memory_manager=self.memory_manager)
        prompt = BENCHMARK_PLANNER_TEMPLATE.format(
            research_area=research_area,
            benchmark_name=benchmark_name,
            evaluation_gap=evaluation_gap,
            construction_approach=construction_approach or "Not yet specified",
            evaluation_framework=evaluation_framework or "Not yet specified",
            data_scale=data_scale or "Not yet specified",
            key_findings=key_findings or "Not yet specified",
            context=context or "No additional context.",
        )

        raw = await client.complete(
            prompt=prompt,
            system=PAPER_PLANNER_SYSTEM_PROMPT,
            task="benchmark_planning",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        provider = getattr(client, "_last_provider", "unknown")

        return self._parse_plan(raw, research_area, "Benchmark", provider)

    def _parse_plan(self, raw: str, research_area: str, paper_type: str, provider: str) -> PaperPlan:
        """Parse LLM output into structured PaperPlan."""
        # Extract positioning
        positioning = ""
        pos_match = re.search(r"(?:positioning|Type)[^\\n]*?:\\s*(.+)", raw, re.IGNORECASE)
        if pos_match:
            positioning = pos_match.group(1).strip()

        # Extract thinking template cells
        template = {}
        for label in ["Research background", "Limitation 1", "Limitation 2", "Limitation 3",
                      "Key Idea / Our Goal", "Challenge 1", "Challenge 2", "Challenge 3",
                      "Methodology topic sentence", "Module A", "Module B", "Module C",
                      "Contribution 1", "Contribution 2", "Contribution 3", "Contribution 4"]:
            pattern = rf"{re.escape(label)}[^\\n]*?\\|([^\\n]+)"
            match = re.search(pattern, raw, re.IGNORECASE)
            if not match:
                pattern = rf"{re.escape(label)}[^\\n]*?:(.+)"
                match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                template[label] = match.group(1).strip(" |\\n")

        # Extract consistency checks
        checks = {}
        for check_name in ["Limitations -> Key Idea", "Key Idea -> Challenges",
                          "Challenges -> Methodology", "Methodology -> Contributions"]:
            pattern = rf"{re.escape(check_name)}[^\\n]*?(pass|fail)"
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                checks[check_name] = match.group(1).lower()

        # Extract gaps
        gaps = []
        for severity in ["CRITICAL", "MAJOR", "MINOR"]:
            pattern = rf"{severity}[^\\n]*?(?:gap|missing|incomplete)[^\\n]*"
            for m in re.finditer(pattern, raw, re.IGNORECASE):
                gaps.append({"cell": "", "severity": severity, "description": m.group(0).strip()})

        # Extract section skeleton
        section_pattern = r"§(\\d+)[^\\n]*?(\\w[^\\n]+)"
        sections = []
        for m in re.finditer(section_pattern, raw):
            sections.append(f"Section {m.group(1)}: {m.group(2).strip()}")

        if not sections:
            for line in raw.split("\n"):
                line = line.strip()
                if line and (line.startswith("Section") or line.startswith("§")):
                    sections.append(line)

        return PaperPlan(
            research_area=research_area,
            paper_type=paper_type,
            positioning=positioning,
            thinking_template=template,
            consistency_checks=checks,
            gaps=gaps[:10],
            section_skeleton=sections[:10] if sections else ["See raw plan for section details."],
            next_skill="intro-drafter" if all(v == "pass" for v in checks.values()) else "(address gaps first)",
            raw_plan=raw,
            planned_at=datetime.now(timezone.utc).isoformat(),
            llm_provider_used=provider,
        )
