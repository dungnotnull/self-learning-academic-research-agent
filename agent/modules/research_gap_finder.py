"""
academic-research-enhanced — ResearchGapFinder
Identifies underexplored research areas via semantic clustering of paper embeddings.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from agent.memory.memory_manager import Paper, ClusterRecord

if TYPE_CHECKING:
    from agent.memory.memory_manager import ResearchMemoryManager

logger = logging.getLogger("academic_agent.research_gap_finder")


@dataclass
class ResearchGap:
    cluster_id: int
    cluster_size: int
    density_score: float
    representative_papers: List[str]  # DOIs
    centroid_keywords: List[str]
    llm_explanation: str = ""
    suggested_directions: List[str] = field(default_factory=list)
    urgency: str = "medium"


@dataclass
class KMeansResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    n_clusters: int
    silhouette: float = 0.0


@dataclass
class GapReport:
    query: str
    n_clusters: int
    gaps: List[ResearchGap]
    clusters: List[Dict]
    density_threshold: float
    paper_count: int


class ResearchGapFinder:
    """Identify underexplored research areas via semantic clustering."""

    def __init__(self, memory_manager: Optional["ResearchMemoryManager"] = None):
        self.memory_manager = memory_manager

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed_papers(self, papers: List[Paper]) -> np.ndarray:
        """Embed paper title+abstract using BGE-large (via HFModelManager)."""
        try:
            from tools.hf_model_manager import HFModelManager
            mgr = HFModelManager.instance()
            texts = [f"{p.title}. {p.abstract[:512]}" for p in papers]
            embeddings = mgr.encode(texts, model_key="bge-large")
            return embeddings
        except Exception as e:
            logger.warning("BGE-large embedding failed: %s; using TF-IDF fallback", e)
            return self._tfidf_fallback(papers)

    def _tfidf_fallback(self, papers: List[Paper]) -> np.ndarray:
        """TF-IDF fallback embeddings when HuggingFace is unavailable."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [f"{p.title} {p.abstract}" for p in papers]
            vectorizer = TfidfVectorizer(max_features=512, stop_words="english")
            mat = vectorizer.fit_transform(texts).toarray()
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms
        except Exception as e:
            logger.warning("TF-IDF fallback also failed: %s; using random embeddings", e)
            return np.random.randn(len(papers), 64)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _select_optimal_k(self, embeddings: np.ndarray, max_k: int = 15) -> int:
        """Select optimal number of clusters using elbow method on inertia."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            return min(8, len(embeddings) // 5 + 2)

        n = len(embeddings)
        if n <= 5:
            return 2
        if n <= 10:
            return min(3, n - 1)

        max_k = min(max_k, n - 1, 15)
        k_range = range(2, max_k + 1)
        inertias = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=3, max_iter=100)
            km.fit(embeddings)
            inertias.append(km.inertia_)

        if len(inertias) < 2:
            return 2

        diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        second_diffs = [diffs[i] - diffs[i + 1] for i in range(len(diffs) - 1)]
        if not second_diffs:
            return int(k_range[0]) + 1

        elbow_idx = second_diffs.index(max(second_diffs))
        optimal_k = int(k_range[elbow_idx + 1])
        logger.debug("Optimal k selected: %d (elbow method)", optimal_k)
        return optimal_k

    def cluster_papers(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> KMeansResult:
        """K-means clustering with cosine distance (via L2 normalization)."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            raise ImportError("scikit-learn is required: pip install scikit-learn")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms

        if n_clusters is None:
            n_clusters = self._select_optimal_k(normalized)

        n_clusters = min(n_clusters, len(embeddings) - 1, 20)
        n_clusters = max(n_clusters, 2)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(normalized)

        sil = 0.0
        if len(embeddings) > n_clusters:
            try:
                sil = float(silhouette_score(normalized, labels, metric="cosine", sample_size=min(1000, len(embeddings))))
            except Exception:
                sil = 0.0

        return KMeansResult(
            labels=labels,
            centroids=km.cluster_centers_,
            inertia=float(km.inertia_),
            n_clusters=n_clusters,
            silhouette=sil,
        )

    def compute_cluster_density(self, cluster_labels: np.ndarray, embeddings: np.ndarray) -> List[float]:
        """Compute intra-cluster cohesion (mean pairwise cosine similarity)."""
        n_clusters = int(cluster_labels.max()) + 1
        densities = []

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = embeddings / norms

        for k in range(n_clusters):
            mask = cluster_labels == k
            cluster_vecs = normalized[mask]

            if len(cluster_vecs) < 2:
                densities.append(1.0)
                continue

            sims = cluster_vecs @ cluster_vecs.T
            n = len(cluster_vecs)
            off_diag_sum = (sims.sum() - n) / (n * (n - 1))
            densities.append(float(max(0.0, min(1.0, off_diag_sum))))

        return densities

    # ------------------------------------------------------------------
    # Gap Identification
    # ------------------------------------------------------------------

    def identify_gaps(
        self,
        papers: List[Paper],
        clusters: KMeansResult,
        densities: List[float],
        threshold: float = 0.45,
    ) -> List[ResearchGap]:
        """Identify low-density clusters as research gaps."""
        gaps = []

        for k in range(clusters.n_clusters):
            density = densities[k]
            if density >= threshold:
                continue

            cluster_indices = [i for i, label in enumerate(clusters.labels) if label == k]
            if len(cluster_indices) < 2:
                continue

            cluster_papers = [papers[i] for i in cluster_indices]

            all_words = []
            for p in cluster_papers:
                words = re.findall(r"\b[a-zA-Z]{4,}\b", p.title.lower())
                all_words.extend(words)

            stop_words = {"this", "that", "with", "from", "have", "been", "using", "based",
                         "deep", "neural", "network", "model", "learning", "paper", "approach"}
            word_freq: Dict[str, int] = {}
            for w in all_words:
                if w not in stop_words:
                    word_freq[w] = word_freq.get(w, 0) + 1

            centroid_keywords = sorted(word_freq, key=lambda x: word_freq[x], reverse=True)[:5]

            if density < 0.25:
                urgency = "high"
            elif density < 0.35:
                urgency = "medium"
            else:
                urgency = "low"

            gaps.append(ResearchGap(
                cluster_id=k,
                cluster_size=len(cluster_indices),
                density_score=density,
                representative_papers=[p.doi for p in cluster_papers[:3]],
                centroid_keywords=centroid_keywords,
                urgency=urgency,
            ))

        gaps.sort(key=lambda g: g.density_score)
        return gaps

    async def explain_gap(
        self, gap: ResearchGap, papers_in_cluster: List[Paper]
    ) -> str:
        """Call LLM to explain the research gap and suggest directions."""
        paper_list = "\n".join(
            f"- {p.title} ({p.year})" for p in papers_in_cluster[:5]
        )
        keywords_str = ", ".join(gap.centroid_keywords)

        prompt = f"""You are a research strategist analyzing an underexplored area.

This cluster of {gap.cluster_size} papers has low semantic cohesion (density score: {gap.density_score:.3f}).
Representative papers:
{paper_list}

Central keywords: {keywords_str}

Analyze and explain:
1. What specific research area does this cluster represent?
2. Why is this area currently underexplored?
3. What are exactly 3 concrete research directions to address this gap?
4. What is the urgency? (high/medium/low) — justify.

Be specific and actionable."""

        try:
            from tools.llm_client import UnifiedLLMClient
            client = UnifiedLLMClient()
            explanation = await client.complete(prompt, task="gap_explanation")
            return explanation
        except Exception as e:
            logger.warning("LLM gap explanation failed: %s", e)
            return (
                f"This cluster represents papers around: {', '.join(gap.centroid_keywords)}. "
                f"The low density ({gap.density_score:.3f}) suggests these {gap.cluster_size} papers "
                f"span multiple subproblems without sufficient cross-pollination. "
                f"Urgency: {gap.urgency}."
            )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def find_gaps(
        self,
        query: str,
        n_clusters: Optional[int] = None,
        density_threshold: float = 0.45,
    ) -> GapReport:
        """Full gap detection pipeline."""
        papers = []
        if self.memory_manager:
            papers = self.memory_manager.search_papers(query, limit=500)

        if len(papers) < 10:
            logger.warning("Too few papers (%d) for clustering", len(papers))
            return GapReport(
                query=query,
                n_clusters=0,
                gaps=[],
                clusters=[],
                density_threshold=density_threshold,
                paper_count=len(papers),
            )

        embeddings = self.embed_papers(papers)
        cluster_result = self.cluster_papers(embeddings, n_clusters)
        densities = self.compute_cluster_density(cluster_result.labels, embeddings)
        gaps = self.identify_gaps(papers, cluster_result, densities, density_threshold)

        for gap in gaps:
            cluster_papers = [
                papers[i]
                for i, label in enumerate(cluster_result.labels)
                if label == gap.cluster_id
            ]
            explanation = await self.explain_gap(gap, cluster_papers)
            gap.llm_explanation = explanation

            lines = explanation.split("\n")
            directions = []
            for line in lines:
                line = line.strip()
                if re.match(r"^[123][\.\)]\s", line):
                    directions.append(re.sub(r"^[123][\.\)]\s*", "", line))
            if directions:
                gap.suggested_directions = directions[:3]

        if self.memory_manager:
            for k in range(cluster_result.n_clusters):
                mask = cluster_result.labels == k
                cluster_papers_dois = [papers[i].doi for i in range(len(papers)) if mask[i]]
                centroid = cluster_result.centroids[k].tolist()
                is_gap = any(g.cluster_id == k for g in gaps)

                record = ClusterRecord(
                    cluster_id=k,
                    topic=query,
                    centroid=centroid,
                    paper_dois=cluster_papers_dois,
                    density=densities[k],
                    is_gap=is_gap,
                    llm_description=next(
                        (g.llm_explanation for g in gaps if g.cluster_id == k), ""
                    ),
                )
                self.memory_manager.save_cluster(record)

        clusters_meta = [
            {
                "cluster_id": k,
                "size": int((cluster_result.labels == k).sum()),
                "density": densities[k],
                "is_gap": any(g.cluster_id == k for g in gaps),
            }
            for k in range(cluster_result.n_clusters)
        ]

        return GapReport(
            query=query,
            n_clusters=cluster_result.n_clusters,
            gaps=gaps,
            clusters=clusters_meta,
            density_threshold=density_threshold,
            paper_count=len(papers),
        )

    def visualize_clusters_text(self, clusters: List[Dict], gaps: List[ResearchGap]) -> str:
        """ASCII text cluster map with gap markers."""
        if not clusters:
            return "No clusters available."

        lines = ["Cluster Map (G = gap):"]
        gap_ids = {g.cluster_id for g in gaps}

        for c in sorted(clusters, key=lambda x: x.get("density", 1.0)):
            cluster_id = c.get("cluster_id", "?")
            size = c.get("size", 0)
            density = c.get("density", 0.0)
            marker = "G" if cluster_id in gap_ids else " "
            bar_len = min(40, int(density * 40))
            bar = "#" * bar_len + "." * (40 - bar_len)
            lines.append(f"  [{marker}] C{cluster_id:02d} ({size:3d} papers) [{bar}] {density:.3f}")

        lines.append(f"\nDensity scale: 0.0 (sparse/gap) -----> 1.0 (dense/explored)")
        lines.append(f"Gaps found: {len(gaps)}/{len(clusters)} clusters")
        return "\n".join(lines)
