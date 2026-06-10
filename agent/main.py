"""
academic-research-enhanced — Agent Entry Point
CLI interface (Click) + FastAPI REST server with authentication, CORS, rate limiting, and file logging.
"""
from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import sys
from typing import List, Optional

import click
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging setup — file rotation + stdout
# ---------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.environ.get("LOG_DIR", "./data/logs")
LOG_FILE = os.path.join(LOG_DIR, "agent.log")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
file_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.getLogger().addHandler(file_handler)

logger = logging.getLogger("academic_agent")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Academic Research Discovery Agent",
    description="Autonomous research paper discovery, citation analysis, gap detection, and literature synthesis.",
    version="1.0.0",
)

# CORS middleware
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrator = None  # lazy global

# ---------------------------------------------------------------------------
# Rate limiting (in-memory token bucket per IP)
# ---------------------------------------------------------------------------
import time
import threading

_rate_lock = threading.Lock()
_rate_buckets: dict[str, list[float]] = {}
RATE_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
RATE_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))


def _check_rate_limit(client_ip: str) -> bool:
    with _rate_lock:
        now = time.time()
        if client_ip not in _rate_buckets:
            _rate_buckets[client_ip] = []
        timestamps = _rate_buckets[client_ip]
        cutoff = now - RATE_WINDOW
        _rate_buckets[client_ip] = [t for t in timestamps if t > cutoff]
        if len(_rate_buckets[client_ip]) >= RATE_REQUESTS:
            return False
        _rate_buckets[client_ip].append(now)
        return True


# ---------------------------------------------------------------------------
# API Key authentication middleware
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("AGENT_API_KEY", "")


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    # Health and metrics always accessible
    if request.url.path in ("/health", "/metrics", "/docs", "/openapi.json", "/redoc"):
        response = await call_next(request)
        return response

    # API key check
    if API_KEY:
        provided = request.headers.get("X-API-Key", "")
        if provided != API_KEY:
            return Response(
                content=json.dumps({"detail": "Invalid or missing API key. Provide X-API-Key header."}),
                status_code=401,
                media_type="application/json",
            )

    # Rate limiting on write endpoints
    write_prefixes = ("/api/v1/crawl", "/api/v1/synthesize", "/api/v1/gaps",
                      "/api/v1/knowledge", "/api/v1/embeddings")
    if request.url.path.startswith(write_prefixes):
        client_ip = request.client.host if request.client else "unknown"
        if not _check_rate_limit(client_ip):
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded. Try again later."}),
                status_code=429,
                media_type="application/json",
            )

    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Orchestrator lazy init
# ---------------------------------------------------------------------------
def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from agent.orchestrator import ResearchOrchestrator
        _orchestrator = ResearchOrchestrator()
    return _orchestrator


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class CrawlRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    sources: List[str] = Field(
        default=["arxiv", "semantic_scholar"],
        description="Sources to crawl: arxiv, semantic_scholar, pubmed, ssrn",
    )
    max_results: int = Field(default=50, ge=1, le=200)
    days_back: int = Field(default=7, ge=1, le=365)


class CrawlResponse(BaseModel):
    query: str
    papers_found: int
    papers_new: int
    sources_queried: List[str]
    sample_titles: List[str]


class AnalyzeRequest(BaseModel):
    topic: str = Field(..., description="Topic label for citation analysis")
    min_citations: int = Field(default=5, ge=0)


class AnalyzeResponse(BaseModel):
    topic: str
    graph_nodes: int
    graph_edges: int
    top_influential: List[dict]
    bridge_papers: List[dict]


class GapsRequest(BaseModel):
    topic: str = Field(..., description="Research topic to analyze for gaps")
    n_clusters: Optional[int] = Field(default=None, ge=2, le=20)


class GapsResponse(BaseModel):
    topic: str
    n_clusters: int
    gaps_found: int
    gaps: List[dict]
    cluster_summary: str


class SynthesizeRequest(BaseModel):
    query: str = Field(..., description="Research question / topic")
    max_papers: int = Field(default=15, ge=3, le=50)
    style: str = Field(default="academic", description="academic|technical|survey|executive")


class SynthesizeResponse(BaseModel):
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
    references: List[dict]
    quality_score: float
    paper_count: int
    llm_provider_used: str


class KnowledgeUpdateResponse(BaseModel):
    papers_added: int
    next_scheduled_run: str
    brain_file_updated: bool


class PapersListResponse(BaseModel):
    total: int
    papers: List[dict]


class CostResponse(BaseModel):
    total_cost_usd: float
    by_provider: dict
    by_task: dict
    total_calls: int


class StatusResponse(BaseModel):
    papers_total: int
    citations_total: int
    clusters_total: int
    syntheses_cached: int
    knowledge_hashes: int
    agent_version: str


class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", min_length=1, max_length=100)
    model_key: str = Field(default="bge-large", description="Model key: bge-large or minilm")


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int
    count: int


class BatchEmbeddingRequest(BaseModel):
    dois: List[str] = Field(..., description="List of paper DOIs to embed", min_length=1, max_length=100)
    model_key: str = Field(default="bge-large", description="Model key: bge-large or minilm")


class BatchEmbeddingResponse(BaseModel):
    embeddings: List[dict]
    model: str
    dimensions: int
    count: int


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "academic-research-agent", "version": "1.0.0"}


@app.post("/api/v1/crawl", response_model=CrawlResponse)
async def crawl_endpoint(req: CrawlRequest):
    orch = _get_orchestrator()
    try:
        result = await orch.crawl_papers(
            query=req.query,
            sources=req.sources,
            max_results=req.max_results,
            days_back=req.days_back,
        )
        return CrawlResponse(**result)
    except Exception as exc:
        logger.exception("Crawl failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(req: AnalyzeRequest):
    orch = _get_orchestrator()
    try:
        result = await orch.analyze_citations(
            topic=req.topic,
            min_citations=req.min_citations,
        )
        return AnalyzeResponse(**result)
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/gaps", response_model=GapsResponse)
async def gaps_endpoint(req: GapsRequest):
    orch = _get_orchestrator()
    try:
        result = await orch.find_gaps(topic=req.topic, n_clusters=req.n_clusters)
        return GapsResponse(**result)
    except Exception as exc:
        logger.exception("Gap detection failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/synthesize", response_model=SynthesizeResponse)
async def synthesize_endpoint(req: SynthesizeRequest):
    orch = _get_orchestrator()
    try:
        result = await orch.synthesize_literature(
            query=req.query,
            max_papers=req.max_papers,
            style=req.style,
        )
        return SynthesizeResponse(**result)
    except Exception as exc:
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/knowledge/update", response_model=KnowledgeUpdateResponse)
async def knowledge_update_endpoint():
    orch = _get_orchestrator()
    try:
        result = await orch.update_knowledge_base()
        return KnowledgeUpdateResponse(**result)
    except Exception as exc:
        logger.exception("Knowledge update failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/papers", response_model=PapersListResponse)
async def papers_list_endpoint(
    topic: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    orch = _get_orchestrator()
    try:
        mm = orch.memory_manager
        papers = mm.search_papers(topic or "", limit=limit)
        return PapersListResponse(
            total=mm.count_papers(),
            papers=[
                {
                    "doi": p.doi,
                    "title": p.title,
                    "year": p.year,
                    "source": p.source,
                    "citation_count": p.citation_count,
                    "score": p.score,
                    "url": p.url,
                }
                for p in papers
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/cost", response_model=CostResponse)
async def cost_endpoint():
    orch = _get_orchestrator()
    result = orch.get_cost_report()
    return CostResponse(**result)


@app.get("/api/v1/status", response_model=StatusResponse)
async def status_endpoint():
    orch = _get_orchestrator()
    s = orch.get_status()
    return StatusResponse(**s)


@app.post("/api/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings_endpoint(req: EmbeddingRequest):
    orch = _get_orchestrator()
    try:
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager.instance()
        embeddings = mgr.encode(req.texts, model_key=req.model_key)
        model_name = HFModelManager.MODEL_REGISTRY.get(req.model_key, req.model_key)
        return EmbeddingResponse(
            embeddings=embeddings.tolist(),
            model=model_name,
            dimensions=embeddings.shape[1] if embeddings.ndim > 1 else len(embeddings[0]),
            count=len(req.texts),
        )
    except Exception as exc:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/embeddings/batch", response_model=BatchEmbeddingResponse)
async def embeddings_batch_endpoint(req: BatchEmbeddingRequest):
    orch = _get_orchestrator()
    try:
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager.instance()
        mm = orch.memory_manager
        results = []
        for doi in req.dois:
            paper_dict = mm.get_paper(doi)
            if paper_dict and paper_dict.get("abstract"):
                text = f"{paper_dict['title']}. {paper_dict['abstract'][:512]}"
            elif paper_dict and paper_dict.get("title"):
                text = paper_dict["title"]
            else:
                text = doi
            emb = mgr.encode([text], model_key=req.model_key)
            results.append({
                "doi": doi,
                "embedding": emb[0].tolist() if emb.ndim > 1 else emb.tolist()[0],
                "dimensions": emb.shape[1] if emb.ndim > 1 else len(emb[0]),
            })
        model_name = HFModelManager.MODEL_REGISTRY.get(req.model_key, req.model_key)
        dims = results[0]["dimensions"] if results else 0
        return BatchEmbeddingResponse(
            embeddings=results,
            model=model_name,
            dimensions=dims,
            count=len(results),
        )
    except Exception as exc:
        logger.exception("Batch embedding failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics", response_class=Response)
async def metrics_endpoint():
    orch = _get_orchestrator()
    status = orch.get_status()
    cost_report = orch.get_cost_report()
    lines = [
        "# HELP papers_crawled_total Total papers ingested",
        "# TYPE papers_crawled_total counter",
        f"papers_crawled_total {status.get('papers_total', 0)}",
        "# HELP citations_total Total citation edges",
        "# TYPE citations_total counter",
        f"citations_total {status.get('citations_total', 0)}",
        "# HELP syntheses_generated_total Total synthesis calls",
        "# TYPE syntheses_generated_total counter",
        f"syntheses_generated_total {status.get('syntheses_cached', 0)}",
        "# HELP gaps_found_total Total research gaps identified",
        "# TYPE gaps_found_total counter",
        f"gaps_found_total {status.get('clusters_total', 0)}",
        "# HELP llm_cost_usd_total Total LLM API cost in USD",
        "# TYPE llm_cost_usd_total counter",
        f"llm_cost_usd_total {cost_report.get('total_cost_usd', 0):.6f}",
        "# HELP llm_calls_total Total LLM API calls",
        "# TYPE llm_calls_total counter",
        f"llm_calls_total {cost_report.get('total_calls', 0)}",
        "# HELP agent_info Agent version info",
        "# TYPE agent_info gauge",
        f'agent_info{{version="1.0.0"}} 1',
    ]

    try:
        from agent.orchestrator import llm_tokens_used_total
        if llm_tokens_used_total is not None:
            lines.append("# HELP llm_tokens_used_total Total LLM tokens used")
            lines.append("# TYPE llm_tokens_used_total counter")
            lines.append(f"llm_tokens_used_total {llm_tokens_used_total._value.get()}")
    except Exception:
        pass

    lines.append("")
    return Response(content="\n".join(lines), media_type="text/plain")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable DEBUG logging.")
def cli(verbose: bool):
    """Academic Research Discovery & Daily Self-Learning Agent."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option("--query", "-q", required=True, help="Search query")
@click.option(
    "--sources",
    "-s",
    default="arxiv,semantic_scholar",
    show_default=True,
    help="Comma-separated sources: arxiv,semantic_scholar,pubmed,ssrn",
)
@click.option("--max-results", default=50, show_default=True, help="Max results per source")
@click.option("--days-back", default=7, show_default=True, help="Days to look back")
def crawl(query: str, sources: str, max_results: int, days_back: int):
    """Crawl papers from all specified sources for a given query."""
    source_list = [s.strip() for s in sources.split(",")]
    orch = _get_orchestrator()

    async def _run():
        result = await orch.crawl_papers(
            query=query,
            sources=source_list,
            max_results=max_results,
            days_back=days_back,
        )
        click.echo(f"\nQuery: {query}")
        click.echo(f"Sources: {', '.join(result['sources_queried'])}")
        click.echo(f"Papers found: {result['papers_found']}")
        click.echo(f"New papers saved: {result['papers_new']}")
        if result.get("sample_titles"):
            click.echo("\nSample titles:")
            for t in result["sample_titles"][:5]:
                click.echo(f"  - {t}")

    asyncio.run(_run())


@cli.command()
@click.option("--topic", "-t", required=True, help="Topic label for citation analysis")
@click.option("--min-citations", default=5, show_default=True, help="Minimum citation count")
def analyze(topic: str, min_citations: int):
    """Run citation network analysis on crawled papers."""
    orch = _get_orchestrator()

    async def _run():
        result = await orch.analyze_citations(topic=topic, min_citations=min_citations)
        click.echo(f"\nCitation Network — {topic}")
        click.echo(f"  Nodes: {result['graph_nodes']}")
        click.echo(f"  Edges: {result['graph_edges']}")
        click.echo("\nTop Influential Papers:")
        for i, p in enumerate(result["top_influential"][:5], 1):
            click.echo(f"  {i}. [{p.get('influence_tier', '?')}] {p['title']} ({p.get('year', '?')})")
            click.echo(f"     Citations: {p.get('citation_count', 0)} | PageRank: {p.get('pagerank_score', 0):.4f}")

    asyncio.run(_run())


@cli.command()
@click.option("--topic", "-t", required=True, help="Research topic to analyze for gaps")
@click.option("--n-clusters", default=None, type=int, help="Number of clusters (auto if not set)")
def gaps(topic: str, n_clusters: Optional[int]):
    """Find research gaps in a topic area via semantic clustering."""
    orch = _get_orchestrator()

    async def _run():
        result = await orch.find_gaps(topic=topic, n_clusters=n_clusters)
        click.echo(f"\nResearch Gap Analysis — {topic}")
        click.echo(f"  Clusters analyzed: {result['n_clusters']}")
        click.echo(f"  Gaps identified: {result['gaps_found']}")
        click.echo()
        for gap in result["gaps"]:
            urgency = gap.get("urgency", "unknown")
            color = {"high": "red", "medium": "yellow", "low": "green"}.get(urgency, "white")
            click.secho(f"  GAP [{urgency.upper()}]: Cluster {gap.get('cluster_id', '?')} ({gap.get('cluster_size', 0)} papers)", fg=color)
            click.echo(f"  Density: {gap.get('density_score', 0):.3f}")
            click.echo(f"  {gap.get('llm_explanation', '')[:200]}...")
            click.echo()

    asyncio.run(_run())


@cli.command()
@click.option("--query", "-q", required=True, help="Research question / topic")
@click.option("--max-papers", default=15, show_default=True, help="Max papers to include")
@click.option(
    "--style",
    default="academic",
    show_default=True,
    type=click.Choice(["academic", "technical", "survey", "executive"]),
    help="Review style",
)
@click.option("--output", "-o", default=None, help="Output file path (default: stdout)")
def synthesize(query: str, max_papers: int, style: str, output: Optional[str]):
    """Generate a literature review for a research query."""
    orch = _get_orchestrator()

    async def _run():
        result = await orch.synthesize_literature(
            query=query,
            max_papers=max_papers,
            style=style,
        )
        review_text = _format_review(result)
        if output:
            with open(output, "w", encoding="utf-8") as fh:
                fh.write(review_text)
            click.echo(f"Literature review saved to: {output}")
        else:
            click.echo(review_text)

    asyncio.run(_run())


def _format_review(result: dict) -> str:
    lines = [
        f"# Literature Review: {result['query']}",
        f"*Generated: {result['generated_at']} | Style: {result['style']} | Papers: {result['paper_count']}*",
        f"*Quality Score: {result.get('quality_score', 0):.2f}/1.0*",
        "",
        "## Introduction",
        result.get("introduction", ""),
        "",
        "## Key Themes",
        "\n".join(f"- {t}" for t in result.get("key_themes", [])),
        "",
        "## Methodology Landscape",
        result.get("methodology_landscape", ""),
        "",
        "## State of the Art",
        result.get("state_of_the_art", ""),
        "",
        "## Identified Gaps",
        "\n".join(f"- {g}" for g in result.get("identified_gaps", [])),
        "",
        "## Future Directions",
        result.get("future_directions", ""),
        "",
        "## Conclusion",
        result.get("conclusion", ""),
        "",
        "## References",
    ]
    for ref in result.get("references", []):
        lines.append(f"[{ref.get('index', '?')}] {ref.get('authors', '')} ({ref.get('year', '?')}). {ref.get('title', '')}. {ref.get('url', '')}")
    return "\n".join(lines)


@cli.command("update-knowledge")
def update_knowledge():
    """Manually trigger SECOND-KNOWLEDGE-BRAIN.md update."""
    orch = _get_orchestrator()

    async def _run():
        result = await orch.update_knowledge_base()
        click.echo(f"Papers added: {result['papers_added']}")
        click.echo(f"Brain file updated: {result['brain_file_updated']}")
        click.echo(f"Next scheduled run: {result['next_scheduled_run']}")

    asyncio.run(_run())


@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind host")
@click.option("--port", default=8018, show_default=True, help="Bind port")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev mode)")
@click.option("--start-scheduler", is_flag=True, default=True, show_default=True, help="Start daily cron scheduler")
def serve(host: str, port: int, reload: bool, start_scheduler: bool):
    """Start FastAPI REST server on specified host:port."""
    click.echo(f"Starting Academic Research Agent server on {host}:{port}")

    if start_scheduler:
        orch = _get_orchestrator()
        orch.start_scheduler()
        click.echo("Daily self-update scheduler started (06:00 daily)")

    uvicorn.run(
        "agent.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
    )


@cli.command("cost-report")
def cost_report():
    """Show LLM API cost summary."""
    orch = _get_orchestrator()
    report = orch.get_cost_report()
    click.echo("\nLLM Cost Report")
    click.echo("=" * 40)
    click.echo(f"Total cost: ")
    click.echo(f"Total calls: {report['total_calls']}")
    click.echo("\nBy provider:")
    for provider, data in report.get("by_provider", {}).items():
        click.echo(f"  {provider}:  ({data.get('calls', 0)} calls)")
    click.echo("\nBy task:")
    for task, data in report.get("by_task", {}).items():
        click.echo(f"  {task}: ")


@cli.command()
def status():
    """Show database statistics (papers, citations, clusters)."""
    orch = _get_orchestrator()
    s = orch.get_status()
    click.echo("\nAgent Status")
    click.echo("=" * 40)
    click.echo(f"Papers indexed:      {s.get('papers_total', 0):,}")
    click.echo(f"Citation edges:      {s.get('citations_total', 0):,}")
    click.echo(f"Clusters stored:     {s.get('clusters_total', 0):,}")
    click.echo(f"Syntheses cached:    {s.get('syntheses_cached', 0):,}")
    click.echo(f"Knowledge hashes:    {s.get('knowledge_hashes', 0):,}")
    click.echo(f"Agent version:       {s.get('agent_version', '1.0.0')}")


@cli.command()
@click.option("--texts", "-t", required=True, help="Comma-separated texts to embed")
@click.option("--model-key", default="bge-large", show_default=True, help="Model key: bge-large or minilm")
def embed(texts: str, model_key: str):
    """Generate embeddings for given texts."""
    from tools.hf_model_manager import HFModelManager
    mgr = HFModelManager.instance()
    text_list = [t.strip() for t in texts.split(",") if t.strip()]
    click.echo(f"Generating embeddings for {len(text_list)} texts using {model_key}...")
    embeddings = mgr.encode(text_list, model_key=model_key)
    for i, (text, emb) in enumerate(zip(text_list, embeddings)):
        click.echo(f"\n[{i+1}] \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        click.echo(f"    Dimensions: {len(emb)}, Norm: {sum(x**2 for x in emb)**0.5:.4f}")


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

def main():
    cli()


if __name__ == "__main__":
    main()
