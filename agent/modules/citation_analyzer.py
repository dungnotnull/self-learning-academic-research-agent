"""
academic-research-enhanced — CitationAnalyzer
Builds citation network, computes PageRank, node2vec embeddings, identifies influential papers.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from agent.memory.memory_manager import Paper

if TYPE_CHECKING:
    from agent.memory.memory_manager import ResearchMemoryManager

logger = logging.getLogger("academic_agent.citation_analyzer")


@dataclass
class InfluentialPaper:
    doi: str
    title: str
    authors: List[str]
    year: int
    pagerank_score: float
    citation_count: int
    betweenness: float = 0.0
    influence_tier: str = "cited"


class CitationAnalyzer:
    """Builds and analyzes citation networks."""

    S2_BASE = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    # ------------------------------------------------------------------
    # Graph Construction
    # ------------------------------------------------------------------

    async def build_citation_graph(self, papers: List[Paper]) -> Any:
        """Build a NetworkX DiGraph from Semantic Scholar citation edges."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx is required: pip install networkx")

        graph = nx.DiGraph()

        for p in papers:
            graph.add_node(
                p.doi,
                title=p.title,
                year=p.year,
                authors=p.authors,
                citation_count=p.citation_count,
            )

        s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        fetch_failed = False

        doi_to_paper = {p.doi: p for p in papers}

        for paper in papers:
            try:
                cited_dois = await self.fetch_citations(paper.doi, s2_key)
                for cited_doi in cited_dois:
                    if cited_doi in doi_to_paper:
                        graph.add_edge(paper.doi, cited_doi)
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug("Could not fetch citations for %s: %s", paper.doi, e)
                fetch_failed = True

        if fetch_failed and graph.number_of_edges() < 3:
            logger.info("Using keyword co-occurrence fallback for citation graph")
            self._add_cooccurrence_edges(graph, papers)

        logger.info(
            "Citation graph built: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph

    async def fetch_citations(self, paper_doi: str, api_key: str = "") -> List[str]:
        """Fetch cited papers from Semantic Scholar API."""
        import aiohttp

        s2_id = paper_doi
        if paper_doi.startswith("arxiv:"):
            s2_id = "arXiv:" + paper_doi[6:]
        elif paper_doi.startswith("s2:"):
            s2_id = paper_doi[3:]

        url = f"{self.S2_BASE}/paper/{s2_id}/references"
        headers = {"x-api-key": api_key} if api_key else {}
        params = {"fields": "externalIds,title", "limit": 50}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()

            dois = []
            for ref in data.get("data", []):
                cited_paper = ref.get("citedPaper", {})
                ext_ids = cited_paper.get("externalIds") or {}
                doi = ext_ids.get("DOI") or (
                    f"arxiv:{ext_ids['ArXiv']}" if ext_ids.get("ArXiv") else ""
                )
                if doi:
                    dois.append(doi)
            return dois

        except Exception as e:
            logger.debug("fetch_citations failed for %s: %s", paper_doi, e)
            return []

    def _add_cooccurrence_edges(self, graph: Any, papers: List[Paper]):
        """Add edges based on keyword overlap in abstracts."""
        import re
        from itertools import combinations

        keyword_sets = {}
        for p in papers:
            words = set(re.findall(r"\b[a-z]{4,}\b", f"{p.title} {p.abstract}".lower()))
            keyword_sets[p.doi] = words

        doi_list = list(keyword_sets.keys())
        for i, j in combinations(range(len(doi_list)), 2):
            doi_a, doi_b = doi_list[i], doi_list[j]
            set_a, set_b = keyword_sets[doi_a], keyword_sets[doi_b]
            if not set_a or not set_b:
                continue
            jaccard = len(set_a & set_b) / len(set_a | set_b)
            if jaccard > 0.15:
                graph.add_edge(doi_a, doi_b)

    # ------------------------------------------------------------------
    # Graph Metrics
    # ------------------------------------------------------------------

    def compute_pagerank(self, graph: Any) -> Dict[str, float]:
        """Compute PageRank scores."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required")

        if graph.number_of_nodes() < 2:
            return {n: 1.0 / max(graph.number_of_nodes(), 1) for n in graph.nodes()}

        try:
            return nx.pagerank(graph, alpha=0.85, max_iter=200)
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge; returning degree-based scores")
            degrees = dict(graph.degree())
            total = sum(degrees.values()) or 1
            return {n: v / total for n, v in degrees.items()}

    def compute_betweenness(self, graph: Any) -> Dict[str, float]:
        """Compute betweenness centrality (normalized)."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx required")

        if graph.number_of_nodes() < 10:
            return {n: 0.0 for n in graph.nodes()}

        try:
            return nx.betweenness_centrality(graph, normalized=True)
        except Exception as e:
            logger.warning("Betweenness centrality failed: %s", e)
            return {n: 0.0 for n in graph.nodes()}

    def identify_influential(self, graph: Any, top_k: int = 20) -> List[InfluentialPaper]:
        """Identify top-k most influential papers by PageRank + citation count."""
        pagerank = self.compute_pagerank(graph)
        betweenness = self.compute_betweenness(graph)

        citation_counts = {n: graph.nodes[n].get("citation_count", 0) for n in graph.nodes()}
        max_citations = max(citation_counts.values()) or 1

        scored = []
        for doi, pr in pagerank.items():
            node_data = graph.nodes[doi]
            norm_citations = citation_counts[doi] / max_citations
            combined_score = pr * 0.6 + norm_citations * 0.4

            if pr > 0.05 or citation_counts[doi] > 100:
                tier = "seminal"
            elif pr > 0.02 or citation_counts[doi] > 20:
                tier = "highly_cited"
            elif betweenness.get(doi, 0) > 0.1:
                tier = "bridge"
            else:
                tier = "cited"

            scored.append(InfluentialPaper(
                doi=doi,
                title=node_data.get("title", ""),
                authors=node_data.get("authors", []),
                year=node_data.get("year", 0),
                pagerank_score=pr,
                citation_count=citation_counts[doi],
                betweenness=betweenness.get(doi, 0.0),
                influence_tier=tier,
            ))

        scored.sort(key=lambda x: x.pagerank_score, reverse=True)
        return scored[:top_k]

    def identify_bridge_papers(self, graph: Any, top_k: int = 5) -> List[InfluentialPaper]:
        """Identify bridge papers (high betweenness centrality)."""
        pagerank = self.compute_pagerank(graph)
        betweenness = self.compute_betweenness(graph)

        bridge_scored = []
        for doi, btw in betweenness.items():
            if btw <= 0:
                continue
            node_data = graph.nodes[doi]
            bridge_scored.append(InfluentialPaper(
                doi=doi,
                title=node_data.get("title", ""),
                authors=node_data.get("authors", []),
                year=node_data.get("year", 0),
                pagerank_score=pagerank.get(doi, 0.0),
                citation_count=node_data.get("citation_count", 0),
                betweenness=btw,
                influence_tier="bridge",
            ))

        bridge_scored.sort(key=lambda x: x.betweenness, reverse=True)
        return bridge_scored[:top_k]

    # ------------------------------------------------------------------
    # node2vec Embeddings
    # ------------------------------------------------------------------

    def embed_papers_node2vec(self, graph: Any) -> Dict[str, np.ndarray]:
        """Compute node2vec embeddings for each paper node."""
        try:
            from node2vec import Node2Vec
        except ImportError:
            logger.warning("node2vec not installed; using degree-based fallback embeddings")
            return self._degree_embedding_fallback(graph)

        try:
            node2vec = Node2Vec(
                graph,
                dimensions=128,
                walk_length=30,
                num_walks=200,
                p=1,
                q=0.5,
                workers=1,
                quiet=True,
            )
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            embeddings = {}
            for node in graph.nodes():
                try:
                    embeddings[node] = np.array(model.wv[node])
                except KeyError:
                    embeddings[node] = np.zeros(128)
            return embeddings
        except Exception as e:
            logger.warning("node2vec failed: %s; using fallback", e)
            return self._degree_embedding_fallback(graph)

    def _degree_embedding_fallback(self, graph: Any) -> Dict[str, np.ndarray]:
        """Degree-based fallback when node2vec is unavailable."""
        try:
            import networkx as nx
        except ImportError:
            return {}

        pr = self.compute_pagerank(graph)
        btw = self.compute_betweenness(graph)
        in_deg = dict(graph.in_degree())
        out_deg = dict(graph.out_degree())
        max_in = max(in_deg.values()) or 1
        max_out = max(out_deg.values()) or 1

        embeddings = {}
        for node in graph.nodes():
            vec = np.array([
                pr.get(node, 0.0),
                btw.get(node, 0.0),
                in_deg.get(node, 0) / max_in,
                out_deg.get(node, 0) / max_out,
            ])
            embeddings[node] = vec
        return embeddings

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def save_graph_snapshot(self, graph: Any, topic: str = "default"):
        """Pickle graph snapshot for later use."""
        if self.memory_manager:
            try:
                data_dir = getattr(self.memory_manager, "data_dir", "./data")
                os.makedirs(data_dir, exist_ok=True)
                snapshot_path = os.path.join(data_dir, f"graph_{topic[:30]}.pkl")
                with open(snapshot_path, "wb") as fh:
                    pickle.dump(graph, fh)
                logger.info("Graph snapshot saved to %s", snapshot_path)
            except Exception as e:
                logger.warning("Failed to save graph snapshot: %s", e)
