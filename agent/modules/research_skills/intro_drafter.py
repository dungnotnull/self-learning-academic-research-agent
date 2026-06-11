"""
academic-research-enhanced — IntroDrafter

Drafts a six-paragraph Introduction outline for a technical or benchmark
paper using the Introduction Flowchart thinking model from Supervisor-Skills.

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

logger = logging.getLogger("academic_agent.intro_drafter")


INTRO_DRAFTER_SYSTEM_PROMPT = """You are a senior research advisor drafting Introduction outlines for top-venue papers (NeurIPS, ICML, ICLR, VLDB, SIGMOD, KDD).

Follow the six-paragraph Introduction Flowchart model:

PARAGRAPH 1 — Background and Motivation: Running example. Why the problem matters in the real world.
PARAGRAPH 2 — Limitations of Existing Work: At most three, each framed as "prior work X does not handle Y".
PARAGRAPH 3 — Problem Essence and Our Goal: Hard constraints explicit. In Technique papers, this is a bridge paragraph. In New Problem papers, this IS the contribution.
PARAGRAPH 4 — Key Challenges: At most three, each explaining why naive extension of prior work fails.
PARAGRAPH 5 — Solution Overview: Each module addresses a challenge. One-to-one mapping between Paragraph 4 challenges and Paragraph 5 modules.
PARAGRAPH 6 — Contributions: Three or four numbered bullets. Each maps to a section reference.

Rules:
- Running example from Paragraph 1 must reappear in Paragraph 5 or 6.
- Limitations (Paragraph 2) are at most three and each is specific.
- Challenges (Paragraph 4) are at most three and each explains why naive fails.
- Challenge-to-module mapping is one-to-one.
- Contributions map to section numbers.
- No vague contributions like "extensive experiments" or "thorough analysis".

Output the outline in the structured format.
"""

INTRO_DRAFTER_TEMPLATE = """Draft a six-paragraph Introduction outline for the following paper:

RESEARCH AREA: {research_area}
PAPER TYPE: {paper_type}
TOPIC: {topic}
KEY IDEA OR GOAL: {key_idea}
LIMITATIONS OF PRIOR WORK: {limitations}
CHALLENGES: {challenges}
SOLUTION OVERVIEW: {solution_overview}
CONTRIBUTIONS: {contributions}
RUNNING EXAMPLE (if known): {running_example}

Produce the complete six-paragraph outline with purpose, writing points, and gaps for each paragraph.
"""


@dataclass
class IntroOutline:
    """Structured result of an Introduction outline."""
    research_area: str
    paper_type: str  # "Technique" | "New Problem/Setting"
    type_positioning_rationale: str
    paragraphs: List[Dict[str, Any]]  # [{"number": 1, "title": "Background and Motivation", "purpose": "...", "writing_points": [...], "gaps": [...]}]
    challenge_module_mapping: List[Dict[str, str]]  # [{"challenge": "...", "module": "..."}]
    contribution_section_mapping: List[Dict[str, str]]  # [{"contribution": "...", "section": "..."}]
    consistency_report: Dict[str, str]  # {"running-example loop": "pass", ...}
    severity_summary: Dict[str, int]  # {"CRITICAL": 0, "MAJOR": 1, "MINOR": 2}
    top_three_actions: List[str]
    raw_outline: str
    drafted_at: str
    llm_provider_used: str = "unknown"


class IntroDrafter:
    """Drafts Introduction outlines using the Flowchart model from Supervisor-Skills."""

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    async def draft(
        self,
        research_area: str,
        topic: str,
        paper_type: str = "Technique",
        key_idea: str = "",
        limitations: str = "",
        challenges: str = "",
        solution_overview: str = "",
        contributions: str = "",
        running_example: str = "",
        max_tokens: int = 4000,
    ) -> IntroOutline:
        """Draft a six-paragraph Introduction outline."""
        from tools.llm_client import UnifiedLLMClient

        # Enrich from memory
        if (not limitations or not challenges) and self.memory_manager:
            papers = self.memory_manager.search_papers(topic, limit=10)
            if papers:
                limitations = limitations or "\\n".join(
                    f"- {p.title} ({p.year}): may not address {topic}"
                    for p in papers[:3]
                )

        client = UnifiedLLMClient(memory_manager=self.memory_manager)
        prompt = INTRO_DRAFTER_TEMPLATE.format(
            research_area=research_area,
            paper_type=paper_type,
            topic=topic,
            key_idea=key_idea or "To be determined",
            limitations=limitations or "To be identified",
            challenges=challenges or "To be identified",
            solution_overview=solution_overview or "To be described",
            contributions=contributions or "To be listed",
            running_example=running_example or "Not yet specified — please propose candidates",
        )

        raw = await client.complete(
            prompt=prompt,
            system=INTRO_DRAFTER_SYSTEM_PROMPT,
            task="intro_drafting",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        provider = getattr(client, "_last_provider", "unknown")

        return self._parse_outline(raw, research_area, paper_type, provider)

    def _parse_outline(self, raw: str, research_area: str, paper_type: str, provider: str) -> IntroOutline:
        """Parse LLM output into structured IntroOutline."""
        paragraphs = []
        para_titles = [
            "Background and Motivation",
            "Limitations of Existing Work",
            "Problem Essence and Our Goal",
            "Key Challenges",
            "Solution Overview",
            "Contributions",
        ]
        for i, title in enumerate(para_titles, 1):
            pattern = rf"Paragraph\s+{i}[^\\n]*{re.escape(title)}(.*?)(?=Paragraph\\s+{i+1}|###\\s*[3-9]|Flowchart|$)"
            match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
            content = match.group(1).strip() if match else ""

            # Extract writing points
            writing_points = []
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("-") or line.startswith("*") or line.startswith("•"):
                    writing_points.append(line.lstrip("-*• "))

            # Extract gaps
            gaps = []
            for line in content.split("\n"):
                line = line.strip()
                if "CRITICAL" in line or "MAJOR" in line or "MINOR" in line:
                    gaps.append(line)

            paragraphs.append({
                "number": i,
                "title": title,
                "purpose": "",
                "writing_points": writing_points[:5],
                "gaps": gaps[:3],
            })

        # Extract challenge-module mapping
        c_m_mapping = []
        for i in range(1, 4):
            pattern = rf"Challenge\\s+{i}[^\\n]*→[^\\n]*Module[^\\n]*"
            match = re.search(pattern, raw, re.IGNORECASE)
            if not match:
                pattern = rf"Challenge\\s+{i}[^\\n]*\\->[^\\n]*"
                match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                c_m_mapping.append({"challenge": f"Challenge {i}", "module": match.group(0)})

        # Extract contribution-section mapping
        c_s_mapping = []
        for i in range(1, 5):
            pattern = rf"(?:Contribution|C)\\s*{i}[^\\n]*Section\\s*\\d+"
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                c_s_mapping.append({"contribution": f"Contribution {i}", "section": match.group(0)})

        # Consistency report
        consistency = {}
        for check in ["running-example loop", "limitations-challenges link", "goal-contribution1 link",
                      "challenge-module mapping", "contribution-section mapping"]:
            pattern = rf"{re.escape(check)}[^\\n]*?(pass|fail)"
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                consistency[check] = match.group(1).lower()
            else:
                consistency[check] = "unknown"

        # Severity summary
        critical_count = len(re.findall(r"CRITICAL", raw, re.IGNORECASE))
        major_count = len(re.findall(r"MAJOR", raw, re.IGNORECASE)) - critical_count
        minor_count = len(re.findall(r"MINOR", raw, re.IGNORECASE))

        # Top actions
        actions = []
        action_pattern = rf"(?:Top three|Action|Fix)[^\\n]*?(?:\\d[\\.)][^\\n]+)"
        for m in re.finditer(action_pattern, raw, re.IGNORECASE):
            actions.append(m.group(0).strip())
        if not actions:
            actions = ["Review the full outline for recommended next steps."]

        return IntroOutline(
            research_area=research_area,
            paper_type=paper_type,
            type_positioning_rationale="",
            paragraphs=paragraphs,
            challenge_module_mapping=c_m_mapping,
            contribution_section_mapping=c_s_mapping,
            consistency_report=consistency,
            severity_summary={"CRITICAL": critical_count, "MAJOR": max(0, major_count), "MINOR": minor_count},
            top_three_actions=actions[:3],
            raw_outline=raw,
            drafted_at=datetime.now(timezone.utc).isoformat(),
            llm_provider_used=provider,
        )
