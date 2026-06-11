"""
academic-research-enhanced — Research Skills API schemas
Pydantic models for the six research skill endpoints.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Idea Evaluator
# ---------------------------------------------------------------------------

class IdeaEvalRequest(BaseModel):
    research_area: str = Field(..., description="Research area / domain")
    idea_description: str = Field(..., description="Research idea to evaluate")
    current_stage: str = Field(default="early brainstorming", description="Current stage of the idea")
    weekly_hours: int = Field(default=20, description="Weekly hours available")
    skill_depth: str = Field(default="intermediate", description="Skill depth: beginner, intermediate, advanced")
    strength_profile: str = Field(default="applied", description="Theoretical vs applied strength")
    compute_availability: str = Field(default="single GPU", description="Compute resources available")
    data_access: str = Field(default="public datasets", description="Data access level")
    timeline: str = Field(default="3 months", description="Project timeline")
    prior_work_context: str = Field(default="", description="Prior work context (optional)")


class IdeaEvalResponse(BaseModel):
    idea_summary: str
    paper_type: str
    one_sentence_story: str = ""
    fatal_flaws: List[Dict[str, str]]
    has_critical_flaw: bool
    lifecycle_category: str = ""
    capability_match: str = ""
    dimension_scores: Dict[str, int]
    disruptive_potential: str
    feasibility_risks: Dict[str, Dict[str, str]] = {}
    verdict: str
    top_three_actions: List[str]
    evaluated_at: str
    llm_provider_used: str = "unknown"


# ---------------------------------------------------------------------------
# Paper Planner
# ---------------------------------------------------------------------------

class TechPaperPlanRequest(BaseModel):
    research_area: str = Field(..., description="Research area")
    topic: str = Field(..., description="Research topic")
    key_idea: str = Field(default="", description="Key idea or method")
    limitations: str = Field(default="", description="Limitations of prior work")
    methodology: str = Field(default="", description="Methodology overview")
    contributions: str = Field(default="", description="Expected contributions")
    context: str = Field(default="", description="Additional context")


class BenchmarkPaperPlanRequest(BaseModel):
    research_area: str = Field(..., description="Research area")
    benchmark_name: str = Field(..., description="Benchmark name")
    evaluation_gap: str = Field(..., description="Evaluation blind spot being addressed")
    construction_approach: str = Field(default="", description="Construction pipeline approach")
    evaluation_framework: str = Field(default="", description="Evaluation framework")
    data_scale: str = Field(default="", description="Data scale")
    key_findings: str = Field(default="", description="Key findings or insights")
    context: str = Field(default="", description="Additional context")


class PaperPlanResponse(BaseModel):
    research_area: str
    paper_type: str
    positioning: str
    thinking_template: Dict[str, str]
    consistency_checks: Dict[str, str]
    gaps: List[Dict[str, str]]
    section_skeleton: List[str]
    next_skill: str
    planned_at: str
    llm_provider_used: str = "unknown"


# ---------------------------------------------------------------------------
# Intro Drafter
# ---------------------------------------------------------------------------

class IntroDraftRequest(BaseModel):
    research_area: str = Field(..., description="Research area")
    topic: str = Field(..., description="Paper topic")
    paper_type: str = Field(default="Technique", description="Technique or Benchmark")
    key_idea: str = Field(default="", description="Key idea or goal")
    limitations: str = Field(default="", description="Limitations of prior work")
    challenges: str = Field(default="", description="Key challenges")
    solution_overview: str = Field(default="", description="Solution overview")
    contributions: str = Field(default="", description="Contributions")
    running_example: str = Field(default="", description="Running example")


class IntroDraftResponse(BaseModel):
    research_area: str
    paper_type: str
    type_positioning_rationale: str = ""
    paragraphs: List[Dict] = []
    challenge_module_mapping: List[Dict[str, str]] = []
    contribution_section_mapping: List[Dict[str, str]] = []
    consistency_report: Dict[str, str] = {}
    severity_summary: Dict[str, int] = {}
    top_three_actions: List[str] = []
    drafted_at: str
    llm_provider_used: str = "unknown"


# ---------------------------------------------------------------------------
# Figure Designer
# ---------------------------------------------------------------------------

class FigureDesignRequest(BaseModel):
    research_area: str = Field(..., description="Research area")
    figure_type: str = Field(default="motivated-example", description="motivated-example, solution-overview, or experimental-results")
    method_name: str = Field(default="", description="Method name")
    target_venue: str = Field(default="", description="Target venue")
    communication_goal: str = Field(default="", description="What the figure should communicate")
    current_description: str = Field(default="", description="Current figure description (if any)")
    running_example: str = Field(default="", description="Running example from the Introduction")
    key_results: str = Field(default="", description="Key results to visualize (for experiment figures)")


class FigureDesignResponse(BaseModel):
    research_area: str
    figure_type: str
    paradigm_recommendation: str
    paradigm_rationale: str = ""
    layout_sketch: str
    labelling_guidance: List[str] = []
    tool_suggestion_primary: str = ""
    tool_suggestion_alternative: str = ""
    tool_rationale: str = ""
    audit_results: Dict[str, str] = {}
    severity_summary: Dict[str, int] = {}
    top_three_actions: List[str] = []
    designed_at: str
    llm_provider_used: str = "unknown"


# ---------------------------------------------------------------------------
# Paper Reviewer
# ---------------------------------------------------------------------------

class PaperReviewRequest(BaseModel):
    research_area: str = Field(..., description="Research area")
    paper_title: str = Field(default="", description="Paper title")
    target_venue: str = Field(default="", description="Target venue")
    paper_text: str = Field(..., description="Full paper text to review")


class PaperReviewResponse(BaseModel):
    research_area: str
    paper_title: str
    target_venue: str
    summary: Dict[str, int]
    dimensions: Dict[str, List[Dict[str, str]]]
    banned_vocabulary_findings: List[Dict[str, str]] = []
    em_dash_findings: List[Dict[str, str]] = []
    final_score: float
    submission_recommendation: str
    top_three_fixes: List[str]
    reviewed_at: str
    llm_provider_used: str = "unknown"


# ---------------------------------------------------------------------------
# Research Workflow
# ---------------------------------------------------------------------------

class ResearchWorkflowRequest(BaseModel):
    research_area: str = Field(..., description="Research area")
    current_phase: str = Field(default="mixed", description="coding, figure, writing, or mixed")
    project_description: str = Field(default="", description="Project description")
    weekly_hours: int = Field(default=20, description="Weekly hours available")
    target_venue: str = Field(default="", description="Target venue")
    deadline: str = Field(default="", description="Project deadline")
    current_progress: str = Field(default="", description="Current progress description")


class ResearchWorkflowResponse(BaseModel):
    research_area: str
    primary_phase: str
    secondary_phases: List[str] = []
    behavioural_rules_acknowledged: bool = True
    workflow_table: List[Dict] = []
    tool_recommendations: Dict[str, Dict[str, str]] = {}
    red_line_reminders: List[str] = []
    verification_points: List[str] = []
    ai_disclosure_requirements: str = ""
    planned_at: str
    llm_provider_used: str = "unknown"
