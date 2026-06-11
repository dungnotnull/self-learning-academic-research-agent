"""
academic-research-enhanced — ResearchWorkflow

Guides AI-assisted research across three sub-flows (Vibe Coding, Vibe Figure,
Vibe Writing) with behavioral rules that keep the user in charge of academic
judgment while delegating mechanical work to AI.

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

logger = logging.getLogger("academic_agent.research_workflow")


RESEARCH_WORKFLOW_SYSTEM_PROMPT = """You are a senior research advisor guiding AI-assisted research. You follow the Vibe Research methodology with six behavioural rules:

RULE 1: AI-assisted work is permitted for literature search, organisation, code/debugging support, and language/expression polish.
RULE 2: Research ideas, problems, designs, technical paths, experimental plans, core conclusions, and novelty MUST be the user's own and fully understood.
RULE 3: Every AI-generated or AI-assisted passage must be verified by the user against actual research process, experimental results, and facts.
RULE 4: No fabricated citations; references come from the user's own reading and confirmation.
RULE 5: No academic misconduct, including fabricated data, experimental results, or plagiarism concealment.
RULE 6: Venue or school AI-disclosure requirements must be honoured.

You guide the user through three sub-flows:
- Vibe Coding: Plan Mode, Small Steps, Clear Requirements. Use Cursor, Claude Code, or Codex.
- Vibe Figure: Four-step figure workflow (design paradigm, layout sketch, labelling, audit). Use PowerPoint+Figma for static, Matplotlib+Seaborn for experimental, Gemini for first-draft.
- Vibe Writing: Red-line rules (no AI ghost-writing of core arguments, no fabricated citations, no outsourced scientific judgment). Use Claude or ChatGPT for polish, Grammarly for grammar, Overleaf for LaTeX.

For each phase, recommend:
1. Primary and alternative tools
2. Time allocation
3. User verification checkpoints
4. Common failure patterns and how to avoid them
"""

RESEARCH_WORKFLOW_TEMPLATE = """Plan an AI-assisted research workflow for:

RESEARCH AREA: {research_area}
CURRENT PHASE: {current_phase}  (coding, figure, writing, or mixed)
PROJECT DESCRIPTION: {project_description}
WEEKLY HOURS: {weekly_hours}
TARGET VENUE: {target_venue}
DEADLINE: {deadline}
CURRENT PROGRESS: {current_progress}

Produce a structured workflow plan following the six behavioural rules and phase-specific procedures.
"""


@dataclass
class WorkflowPlan:
    """Structured result of a research workflow plan."""
    research_area: str
    primary_phase: str  # "coding" | "figure" | "writing" | "mixed"
    secondary_phases: List[str]
    behavioural_rules_acknowledged: bool
    workflow_table: List[Dict[str, str]]  # [{time_block, phase, activity, tool, user_check}]
    tool_recommendations: Dict[str, Dict[str, str]]  # {"coding": {"primary": "...", "alternative": "...", "reason": "..."}}
    red_line_reminders: List[str]
    verification_points: List[str]
    ai_disclosure_requirements: str
    raw_plan: str
    planned_at: str
    llm_provider_used: str = "unknown"


class ResearchWorkflow:
    """Guides AI-assisted research workflow using the Vibe Research methodology."""

    RED_LINES = [
        "No fabricated citations — references must come from your own reading and confirmation",
        "No AI ghost-writing of core arguments, novelty claims, or scientific judgment",
        "No fabricated data or experimental results",
        "No plagiarism concealment — AI assistance must be disclosed per venue requirements",
        "Every AI-generated passage must be verified sentence-by-sentence",
        "Research direction, problem framing, and contributions must be owned by the researcher",
    ]

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    async def plan(
        self,
        research_area: str,
        current_phase: str = "mixed",
        project_description: str = "",
        weekly_hours: int = 20,
        target_venue: str = "",
        deadline: str = "",
        current_progress: str = "",
        max_tokens: int = 4000,
    ) -> WorkflowPlan:
        """Generate a research workflow plan."""
        from tools.llm_client import UnifiedLLMClient

        # Enrich from memory
        context = ""
        if self.memory_manager and not project_description:
            papers = self.memory_manager.search_papers(research_area, limit=5)
            if papers:
                context = f"Found {len(papers)} papers in '{research_area}': " + ", ".join(p.title[:60] for p in papers[:5])

        client = UnifiedLLMClient(memory_manager=self.memory_manager)
        prompt = RESEARCH_WORKFLOW_TEMPLATE.format(
            research_area=research_area,
            current_phase=current_phase,
            project_description=project_description or context or "Research project",
            weekly_hours=weekly_hours,
            target_venue=target_venue or "Top venue",
            deadline=deadline or "Not specified",
            current_progress=current_progress or "Early stage",
        )

        raw = await client.complete(
            prompt=prompt,
            system=RESEARCH_WORKFLOW_SYSTEM_PROMPT,
            task="research_workflow",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        provider = getattr(client, "_last_provider", "unknown")

        return self._parse_plan(raw, research_area, current_phase, provider)

    def _parse_plan(self, raw: str, research_area: str, current_phase: str, provider: str) -> WorkflowPlan:
        """Parse LLM output into structured WorkflowPlan."""
        # Extract workflow table
        workflow = []
        for line in raw.split("\n"):
            line = line.strip()
            if "|" in line and ("coding" in line.lower() or "figure" in line.lower() or "writing" in line.lower()):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 3:
                    workflow.append({"description": parts[0], "phase": parts[1] if len(parts) > 1 else "", "details": " | ".join(parts[2:])})

        if not workflow:
            workflow = [
                {"description": "Phase 1: Setup and planning", "phase": current_phase, "details": "Define scope and verify with idea-evaluator"},
                {"description": "Phase 2: Implementation", "phase": "coding", "details": "Use Vibe Coding with Plan Mode"},
                {"description": "Phase 3: Figures", "phase": "figure", "details": "Design and audit three core figures"},
                {"description": "Phase 4: Writing", "phase": "writing", "details": "Draft with intro-drafter, review with paper-reviewer"},
            ]

        # Extract tool recommendations
        tools = {
            "coding": {"primary": "Cursor", "alternative": "Claude Code", "reason": "IDE-native AI coding with Plan Mode"},
            "figure": {"primary": "Figma", "alternative": "Matplotlib", "reason": "Professional static figures with AI polish"},
            "writing": {"primary": "Claude", "alternative": "Grammarly", "reason": "Language polish with red-line rules"},
        }

        # Extract red-line reminders
        red_lines = self.RED_LINES

        # Extract verification points
        verification = [
            "No fabricated citation introduced or accepted",
            "Research direction and contributions are user-owned",
            "Every AI-generated code block has been reviewed and tested",
            "Every AI-drafted paragraph has been verified sentence-by-sentence",
            "Venue AI-disclosure rules have been checked",
        ]

        return WorkflowPlan(
            research_area=research_area,
            primary_phase=current_phase,
            secondary_phases=["coding", "figure", "writing"] if current_phase == "mixed" else [],
            behavioural_rules_acknowledged=True,
            workflow_table=workflow,
            tool_recommendations=tools,
            red_line_reminders=red_lines,
            verification_points=verification,
            ai_disclosure_requirements="Check target venue's AI disclosure policy. Most venues require acknowledging AI assistance in the methodology or acknowledgments section.",
            raw_plan=raw,
            planned_at=datetime.now(timezone.utc).isoformat(),
            llm_provider_used=provider,
        )
