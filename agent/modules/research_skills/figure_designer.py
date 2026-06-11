"""
academic-research-enhanced — FigureDesigner

Advises on the design of three core figures in a technical paper:
Motivated Example (Figure 1), Solution Overview (Methodology),
and Experimental Results. Recommends design paradigms, layout,
labelling, tools, and runs a quality-control audit.

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

logger = logging.getLogger("academic_agent.figure_designer")


FIGURE_DESIGNER_SYSTEM_PROMPT = """You are a senior research advisor advising on scientific figure design for top-venue papers (NeurIPS, ICML, ICLR, VLDB, SIGMOD, KDD).

A top-venue paper typically has three core figures that carry the storytelling weight:
1. Motivated Example (Figure 1, page 1-2): Concrete failure-revealing or contrast example.
2. Solution Overview (Methodology section): Framework diagram showing modules and data flow.
3. Experimental Results (Experiments section): Charts showing quantitative findings.

For each figure, you will:
- Identify the figure type
- Recommend the best design paradigm
- Produce a layout sketch the user could draw from
- Provide labelling and annotation guidance
- Suggest the right tool
- Run a universal rule audit (vector format, font size, colour-blind safe, self-contained caption, honest axes, no chartjunk)

Universal rules:
- One figure tells ONE story
- Vector format (PDF, EPS, SVG) for export
- Font size at least 8pt post-scaling
- Small canvas, not large canvas with small fonts
- Colour-blind-safe palette; no colour-only encoding
- Self-contained caption with a finding in the first sentence
- Honest axis ranges
- No 3D effects, no chartjunk
"""

FIGURE_DESIGNER_TEMPLATE = """Design advice for a paper figure:

RESEARCH AREA: {research_area}
PAPER METHOD NAME: {method_name}
TARGET VENUE: {target_venue}
FIGURE TYPE REQUESTED: {figure_type}
WHAT THE FIGURE SHOULD COMMUNICATE: {communication_goal}
CURRENT FIGURE DESCRIPTION (if any): {current_description}
RUNNING EXAMPLE (if known): {running_example}
KEY RESULTS TO VISUALIZE (for experiment figures): {key_results}

Provide complete design advice following your system instructions.
"""


@dataclass
class FigureDesign:
    """Structured result of figure design advice."""
    research_area: str
    figure_type: str  # "motivated-example" | "solution-overview" | "experimental-results"
    paradigm_recommendation: str
    paradigm_rationale: str
    layout_sketch: str
    labelling_guidance: List[str]
    tool_suggestion_primary: str
    tool_suggestion_alternative: str
    tool_rationale: str
    audit_results: Dict[str, str]  # {"vector_format": "pass", ...}
    severity_summary: Dict[str, int]
    top_three_actions: List[str]
    raw_design: str
    designed_at: str
    llm_provider_used: str = "unknown"


class FigureDesigner:
    """Advises on paper figure design using the Supervisor-Skills methodology."""

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    async def design(
        self,
        research_area: str,
        figure_type: str = "motivated-example",
        method_name: str = "",
        target_venue: str = "",
        communication_goal: str = "",
        current_description: str = "",
        running_example: str = "",
        key_results: str = "",
        max_tokens: int = 4000,
    ) -> FigureDesign:
        """Get figure design advice."""
        from tools.llm_client import UnifiedLLMClient

        client = UnifiedLLMClient(memory_manager=self.memory_manager)
        prompt = FIGURE_DESIGNER_TEMPLATE.format(
            research_area=research_area,
            method_name=method_name or "Not specified",
            target_venue=target_venue or "Top venue",
            figure_type=figure_type,
            communication_goal=communication_goal or "To be determined",
            current_description=current_description or "None",
            running_example=running_example or "Not yet specified",
            key_results=key_results or "Not yet available",
        )

        raw = await client.complete(
            prompt=prompt,
            system=FIGURE_DESIGNER_SYSTEM_PROMPT,
            task="figure_design",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        provider = getattr(client, "_last_provider", "unknown")

        return self._parse_design(raw, research_area, figure_type, provider)

    def _parse_design(self, raw: str, research_area: str, figure_type: str, provider: str) -> FigureDesign:
        """Parse LLM output into structured FigureDesign."""
        # Extract paradigm
        paradigm = ""
        paradigm_match = re.search(r"(?:Paradigm|Recommendation)[^\\n]*?:\\s*(.+)", raw, re.IGNORECASE)
        if paradigm_match:
            paradigm = paradigm_match.group(1).strip()

        # Extract layout sketch
        layout = ""
        layout_match = re.search(r"(?:Layout sketch|Canvas)[^\\n]*\\n(.*?)(?=###|Labelling|$)", raw, re.DOTALL | re.IGNORECASE)
        if layout_match:
            layout = layout_match.group(1).strip()

        # Extract labels
        labels = []
        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("-") and ("label" in line.lower() or "element" in line.lower()):
                labels.append(line.lstrip("-* "))

        # Extract tools
        tools_primary = "Figma"
        tools_alt = "PowerPoint"
        tool_match = re.search(r"Primary[^\\n]*?:\\s*(.+)", raw, re.IGNORECASE)
        if tool_match:
            tools_primary = tool_match.group(1).strip()
        tool_match = re.search(r"Alternative[^\\n]*?:\\s*(.+)", raw, re.IGNORECASE)
        if tool_match:
            tools_alt = tool_match.group(1).strip()

        # Extract audit results
        audit = {}
        for rule in ["vector format", "font size", "colour-blind safe", "self-contained caption",
                     "honest axis", "no chartjunk"]:
            pattern = rf"{re.escape(rule)}[^\\n]*?(pass|fail)"
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                audit[rule] = match.group(1).lower()
            else:
                audit[rule] = "unknown"

        # Severity
        critical = len(re.findall(r"CRITICAL", raw, re.IGNORECASE))
        major = len(re.findall(r"MAJOR", raw, re.IGNORECASE)) - critical
        minor = len(re.findall(r"MINOR", raw, re.IGNORECASE))

        # Actions
        actions = []
        action_section = re.search(r"(?:Top three|actions)[^\\n]*\\n(.*?)(?=###|$)", raw, re.DOTALL | re.IGNORECASE)
        if action_section:
            for line in action_section.group(1).split("\n"):
                line = line.strip()
                if line and re.match(r"^\\d", line):
                    actions.append(re.sub(r"^\\d+[\\.)\\]\\s]*", "", line))
        if not actions:
            actions = ["Review the full design output for recommendations."]

        return FigureDesign(
            research_area=research_area,
            figure_type=figure_type,
            paradigm_recommendation=paradigm,
            paradigm_rationale="",
            layout_sketch=layout[:500] if layout else raw[:500],
            labelling_guidance=labels[:5],
            tool_suggestion_primary=tools_primary,
            tool_suggestion_alternative=tools_alt,
            tool_rationale="",
            audit_results=audit,
            severity_summary={"CRITICAL": critical, "MAJOR": max(0, major), "MINOR": minor},
            top_three_actions=actions[:3],
            raw_design=raw,
            designed_at=datetime.now(timezone.utc).isoformat(),
            llm_provider_used=provider,
        )
