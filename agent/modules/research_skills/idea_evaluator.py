"""
academic-research-enhanced — IdeaEvaluator

Evaluates a preliminary research idea against a five-dimension framework
(Highher, Faster, Stronger, Cheaper, Broader), idea-lifecycle and capability
matching, paradigm-shift probing, and a fatal-flaws audit.

Based on: https://github.com/HKUSTDial/Supervisor-Skills
License: CC BY-NC-SA 4.0 (original); integrated under project MIT license.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.memory.memory_manager import ResearchMemoryManager

logger = logging.getLogger("academic_agent.idea_evaluator")


# ---------------------------------------------------------------------------
# Five-dimension framework prompt templates
# ---------------------------------------------------------------------------

IDEA_EVALUATOR_SYSTEM_PROMPT = """You are a senior research advisor evaluating a research idea from the perspective of a top-venue reviewer. You have extensive experience at NeurIPS, ICML, ICLR, VLDB, SIGMOD, KDD, ACL, EMNLP, CVPR, and similar venues.

Your evaluation follows the Supervisor-Skills methodology:

STEP 1 - FIRST IMPRESSION: State whether the idea reads as Novel Problem, Novel Method, or New Setting. Write a one-sentence story. If you cannot, the idea is not clear enough.

STEP 2 - FATAL-FLAWS AUDIT (Early Gate): Run before scoring. Check for:
  - No real novelty (incremental extension of existing work)
  - Orphan problem (no real users or community demand)
  - Evaluation impossible (no data or metric can verify the claim)
  - Scope too broad or too narrow for a single paper
  - Execution risk (requires unattainable compute or data)
If any flaw is CRITICAL (single-handedly causes rejection, unfixable), verdict is immediately "Reject and Pivot". Do NOT run further steps.

STEP 3 - LIFECYCLE AND CAPABILITY MATCHING: Map the idea onto six lifecycle categories (Application, Foundational Theory, Cross-Disciplinary, Frontier Exploration, Data-Intensive, Innovative Technique). Match against the user's declared capability (hours per week, skill depth, theoretical vs applied strength).

STEP 4 - FIVE-DIMENSION SCORING: Score each dimension 1-10 with explicit evidence:
  - Higher: effectiveness and accuracy gains
  - Faster: efficiency and cost reduction
  - Stronger: robustness, noise tolerance, generalisation
  - Cheaper: data, annotation, or solution cost reduction
  - Broader: cross-domain transplantation or unification

STEP 5 - PARADIGM-SHIFT PROBE: Test four questions:
  1. Does it challenge a hidden assumption the field takes for granted? (First Principles)
  2. Does it address an elephant-in-the-room problem? (Elephant in the Room)
  3. Does it ride a technology-cycle shift? (Technology Cycle)
  4. If this problem solved itself, would the field change meaningfully? (Hamming's Rule)
Two or more yes = disruptive potential.

STEP 6 - FEASIBILITY CHECK: Assess compute risk, data risk, engineering risk, timeline risk against the user's stated resources.

STEP 7 - INTEGRITY GATE: Verify that every score cites specific evidence, no score is gut feeling, and the verdict is consistent with scoring.

STEP 8 - FINAL VERDICT: Issue one of:
  - Strong Accept: execute now (2+ dimensions at 8+, no fatal flaws, capability match green)
  - Accept with Revisions: pivot scope per recommendations before starting
  - Reject and Pivot: do not pursue this version
"""

IDEA_EVALUATOR_USER_TEMPLATE = """Evaluate the following research idea:

RESEARCH AREA: {research_area}
IDEA DESCRIPTION: {idea_description}
CURRENT STAGE: {current_stage}
WEEKLY HOURS AVAILABLE: {weekly_hours}
SKILL DEPTH: {skill_depth}
THEORETICAL VS APPLIED STRENGTH: {strength_profile}
COMPUTE AVAILABILITY: {compute_availability}
DATA ACCESS: {data_access}
TIMELINE: {timeline}

If relevant prior work or context is available:
{prior_work_context}

Produce your evaluation in the structured format defined in your system instructions.
"""


@dataclass
class IdeaEvaluation:
    """Structured result of an idea evaluation."""
    idea_summary: str
    paper_type: str  # "Novel Problem" | "Novel Method" | "New Setting"
    one_sentence_story: str
    fatal_flaws: List[Dict[str, str]]  # [{flaw, severity, defense}]
    has_critical_flaw: bool
    lifecycle_category: str
    capability_match: str  # "Green" | "Yellow" | "Red"
    dimension_scores: Dict[str, int]  # {"Higher": 7, "Faster": 5, ...}
    dimension_evidence: Dict[str, str]
    paradigm_shift_answers: Dict[str, Dict[str, str]]  # {"First Principles": {"answer": "No", "rationale": "..."}}
    disruptive_potential: str  # "none" | "possible" | "strong"
    feasibility_risks: Dict[str, Dict[str, str]]  # {"Compute": {"level": "Low", "mitigation": "..."}}
    verdict: str  # "Strong Accept" | "Accept with Revisions" | "Reject and Pivot"
    top_three_actions: List[str]
    raw_evaluation: str
    evaluated_at: str
    llm_provider_used: str = "unknown"


class IdeaEvaluator:
    """Evaluates research ideas using the five-dimension framework from Supervisor-Skills."""

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    async def evaluate(
        self,
        research_area: str,
        idea_description: str,
        current_stage: str = "early brainstorming",
        weekly_hours: int = 20,
        skill_depth: str = "intermediate",
        strength_profile: str = "applied",
        compute_availability: str = "single GPU",
        data_access: str = "public datasets",
        timeline: str = "3 months",
        prior_work_context: str = "",
        max_tokens: int = 4000,
    ) -> IdeaEvaluation:
        """Run the full idea evaluation pipeline."""
        from tools.llm_client import UnifiedLLMClient

        client = UnifiedLLMClient(memory_manager=self.memory_manager)

        # Gather prior work context from memory if available
        if not prior_work_context and self.memory_manager:
            papers = self.memory_manager.search_papers(research_area, limit=10)
            if papers:
                prior_work_context = "\\n".join(
                    f"- {p.title} ({p.year}, cited {p.citation_count}x): {p.abstract[:200]}..."
                    for p in papers[:10]
                )

        prompt = IDEA_EVALUATOR_USER_TEMPLATE.format(
            research_area=research_area,
            idea_description=idea_description,
            current_stage=current_stage,
            weekly_hours=weekly_hours,
            skill_depth=skill_depth,
            strength_profile=strength_profile,
            compute_availability=compute_availability,
            data_access=data_access,
            timeline=timeline,
            prior_work_context=prior_work_context or "No prior work context available.",
        )

        raw = await client.complete(
            prompt=prompt,
            system=IDEA_EVALUATOR_SYSTEM_PROMPT,
            task="idea_evaluation",
            max_tokens=max_tokens,
            temperature=0.3,
        )
        provider = getattr(client, "_last_provider", "unknown")

        evaluation = self._parse_evaluation(raw, research_area, idea_description, provider)

        # Cache in memory if available
        if self.memory_manager:
            try:
                self._cache_evaluation(evaluation)
            except Exception as e:
                logger.warning("Failed to cache idea evaluation: %s", e)

        return evaluation

    def _parse_evaluation(
        self, raw: str, research_area: str, idea_description: str, provider: str
    ) -> IdeaEvaluation:
        """Parse LLM output into structured IdeaEvaluation."""
        import re

        # Extract verdict
        verdict = "Accept with Revisions"  # default
        for v in ["Strong Accept", "Accept with Revisions", "Reject and Pivot"]:
            if v.lower() in raw.lower():
                verdict = v
                break

        # Extract paper type
        paper_type = "Novel Method"
        for pt in ["Novel Problem", "Novel Method", "New Setting"]:
            if pt.lower() in raw.lower():
                paper_type = pt
                break

        # Extract dimension scores
        dimension_scores = {"Higher": 5, "Faster": 5, "Stronger": 5, "Cheaper": 5, "Broader": 5}
        dimension_evidence = {}
        for dim in ["Higher", "Faster", "Stronger", "Cheaper", "Broader"]:
            pattern = rf"{dim}\D*?(\d{{1,2}})"
            match = re.search(pattern, raw, re.IGNORECASE)
            if match:
                score = min(10, max(1, int(match.group(1))))
                dimension_scores[dim] = score

        # Extract fatal flaws
        fatal_flaws = []
        has_critical = False
        flaw_section = re.search(r"(?:Fatal.flaws|fatal.flaws.audit)(.*?)(?=###\s*[3-8]|STEP|Lifecycle|$)", raw, re.DOTALL | re.IGNORECASE)
        if flaw_section:
            for line in flaw_section.group(1).split("\n"):
                line = line.strip()
                if line and (line.startswith("-") or line.startswith("|") or line.startswith("*")):
                    severity = "MAJOR"
                    if "CRITICAL" in line.upper():
                        severity = "CRITICAL"
                        has_critical = True
                    fatal_flaws.append({"flaw": line.lstrip("-*| "), "severity": severity, "defense": ""})

        # Extract paradigm shift answers
        paradigm_shifts = {}
        for probe in ["First Principles", "Elephant in the Room", "Technology Cycle", "Hamming's Rule"]:
            pattern = rf"{probe}[^\\n]*?(Yes|No)"
            match = re.search(pattern, raw, re.IGNORECASE)
            answer = match.group(1) if match else "No"
            paradigm_shifts[probe] = {"answer": answer, "rationale": ""}

        yes_count = sum(1 for v in paradigm_shifts.values() if v["answer"].lower() == "yes")
        if yes_count >= 2:
            disruptive = "strong"
        elif yes_count == 1:
            disruptive = "possible"
        else:
            disruptive = "none"

        # Extract top actions
        actions = []
        action_match = re.search(r"Top three actions.*?(?:\n|$)(.*?)(?=###|$)", raw, re.DOTALL | re.IGNORECASE)
        if action_match:
            for line in action_match.group(1).split("\n"):
                line = line.strip()
                if line and re.match(r"^[\d\-*]", line):
                    actions.append(re.sub(r"^[\d\-\.\)]\\s*", "", line))
        if not actions:
            actions = ["Review the full evaluation output for recommended next steps."]

        return IdeaEvaluation(
            idea_summary=f"{research_area}: {idea_description[:200]}",
            paper_type=paper_type,
            one_sentence_story="",
            fatal_flaws=fatal_flaws[:5],
            has_critical_flaw=has_critical,
            lifecycle_category="",
            capability_match="",
            dimension_scores=dimension_scores,
            dimension_evidence=dimension_evidence,
            paradigm_shift_answers=paradigm_shifts,
            disruptive_potential=disruptive,
            feasibility_risks={},
            verdict=verdict,
            top_three_actions=actions[:3],
            raw_evaluation=raw,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            llm_provider_used=provider,
        )

    def _cache_evaluation(self, evaluation: IdeaEvaluation):
        """Cache evaluation in the synthesis_cache table for retrieval."""
        if not self.memory_manager:
            return
        import hashlib
        key = hashlib.sha256(evaluation.idea_summary.lower().encode()).hexdigest()
        self.memory_manager.save_synthesis(
            f"idea_eval:{evaluation.idea_summary[:100]}",
            {
                "type": "idea_evaluation",
                "idea_summary": evaluation.idea_summary,
                "paper_type": evaluation.paper_type,
                "verdict": evaluation.verdict,
                "dimension_scores": evaluation.dimension_scores,
                "disruptive_potential": evaluation.disruptive_potential,
                "has_critical_flaw": evaluation.has_critical_flaw,
                "top_three_actions": evaluation.top_three_actions,
                "evaluated_at": evaluation.evaluated_at,
            },
        )
