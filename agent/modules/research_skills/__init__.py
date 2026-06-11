"""
academic-research-enhanced — Research Skills Package

Integrates the Supervisor-Skills methodology (HKUSTDial) into the agent
as structured, LLM-powered research skills that can be invoked via CLI,
REST API, or programmatically from the orchestrator pipeline.

Skills:
  - IdeaEvaluator: Five-dimension research idea evaluation
  - PaperPlanner: Technical + Benchmark paper logical skeleton structuring
  - IntroDrafter: Six-paragraph Introduction outline drafting
  - FigureDesigner: Three core figure design and audit
  - PaperReviewer: Pre-submission five-dimension review
  - ResearchWorkflow: Vibe Research AI-assisted workflow guide

Source: https://github.com/HKUSTDial/Supervisor-Skills (CC BY-NC-SA 4.0)
"""

from agent.modules.research_skills.idea_evaluator import IdeaEvaluator
from agent.modules.research_skills.paper_planner import PaperPlanner
from agent.modules.research_skills.intro_drafter import IntroDrafter
from agent.modules.research_skills.figure_designer import FigureDesigner
from agent.modules.research_skills.paper_reviewer import PaperReviewer
from agent.modules.research_skills.research_workflow import ResearchWorkflow

__all__ = [
    "IdeaEvaluator",
    "PaperPlanner",
    "IntroDrafter",
    "FigureDesigner",
    "PaperReviewer",
    "ResearchWorkflow",
]
