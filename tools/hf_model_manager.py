"""
academic-research-enhanced — HFModelManager
Singleton lazy-loader for HuggingFace models with CUDA auto-detection and idle unload.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("academic_agent.hf_model_manager")

MODEL_REGISTRY: Dict[str, str] = {
    "bge-large":    "BAAI/bge-large-en-v1.5",
    "minilm":       "sentence-transformers/all-MiniLM-L6-v2",
    "bart-cnn":     "facebook/bart-large-cnn",
    "bge-reranker": "BAAI/bge-reranker-large",
}

IDLE_UNLOAD_SECONDS = 600  # 10 minutes


class HFModelManager:
    """Singleton registry for HuggingFace models with lazy loading."""

    _instance: Optional["HFModelManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._model_locks: Dict[str, threading.Lock] = {k: threading.Lock() for k in MODEL_REGISTRY}
        self._idle_timers: Dict[str, threading.Timer] = {}
        self._device = self._detect_device()
        self._transformers_available = self._check_transformers()

    @classmethod
    def instance(cls) -> "HFModelManager":
        """Get singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _detect_device(self) -> str:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("HFModelManager device: %s", device)
            return device
        except ImportError:
            logger.info("PyTorch not available; using CPU")
            return "cpu"

    def _check_transformers(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            logger.warning("transformers not installed; using fallback methods")
            return False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, key: str):
        """Lazy-load a model by registry key."""
        if not self._transformers_available:
            return

        model_id = MODEL_REGISTRY.get(key)
        if model_id is None:
            raise ValueError(f"Unknown model key: {key}")

        with self._model_locks[key]:
            if key in self._models:
                self._reset_idle_timer(key)
                return

            logger.info("Loading model: %s (%s)", key, model_id)
            try:
                if key in ("bge-large", "minilm"):
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(model_id, device=self._device)
                    self._models[key] = model

                elif key == "bge-reranker":
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    import torch
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForSequenceClassification.from_pretrained(model_id)
                    model = model.to(self._device)
                    model.eval()
                    self._models[key] = model
                    self._tokenizers[key] = tokenizer

                elif key == "bart-cnn":
                    from transformers import pipeline
                    # Use pipeline for BART summarization
                    device_id = 0 if self._device == "cuda" else -1
                    pipe = pipeline(
                        "summarization",
                        model=model_id,
                        device=device_id,
                        torch_dtype="auto" if self._device == "cuda" else None,
                    )
                    self._models[key] = pipe

                logger.info("Model loaded: %s", key)
                self._reset_idle_timer(key)

            except Exception as e:
                logger.error("Failed to load model %s: %s", key, e)
                # Don't raise; let callers use fallback

    def _reset_idle_timer(self, key: str):
        """Reset the idle unload timer for a model."""
        if key in self._idle_timers:
            self._idle_timers[key].cancel()
        timer = threading.Timer(IDLE_UNLOAD_SECONDS, self._unload_model, args=[key])
        timer.daemon = True
        timer.start()
        self._idle_timers[key] = timer

    def _unload_model(self, key: str):
        """Unload model from memory after idle timeout."""
        with self._model_locks.get(key, threading.Lock()):
            if key in self._models:
                del self._models[key]
                logger.info("Model unloaded after idle: %s", key)
            if key in self._tokenizers:
                del self._tokenizers[key]

    def preload(self, *model_keys: str):
        """Warm up specified models."""
        for key in model_keys:
            if key in MODEL_REGISTRY:
                self._load_model(key)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, texts: List[str], model_key: str = "bge-large") -> np.ndarray:
        """Encode texts to dense embeddings, L2 normalized."""
        if not texts:
            return np.zeros((0, 1024))

        self._load_model(model_key)
        self._reset_idle_timer(model_key)

        if model_key not in self._models:
            return self._numpy_fallback_encode(texts)

        model = self._models[model_key]
        try:
            from sentence_transformers import SentenceTransformer
            if isinstance(model, SentenceTransformer):
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=32,
                )
                return np.array(embeddings)
        except Exception as e:
            logger.warning("Encoding failed for %s: %s; using fallback", model_key, e)
            return self._numpy_fallback_encode(texts)

        return self._numpy_fallback_encode(texts)

    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        model_key: str = "bge-large",
    ) -> np.ndarray:
        """Encode in chunks for large inputs."""
        if not texts:
            return np.zeros((0, 1024))

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = self.encode(batch, model_key=model_key)
            all_embeddings.append(emb)

        return np.vstack(all_embeddings)

    def _numpy_fallback_encode(self, texts: List[str]) -> np.ndarray:
        """TF-IDF-based fallback encoding when transformers unavailable."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1024, stop_words="english")
            mat = vectorizer.fit_transform(texts).toarray()
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return mat / norms
        except Exception as e:
            logger.warning("TF-IDF fallback failed: %s; returning zeros", e)
            return np.zeros((len(texts), 128))

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """Rerank passages against query using BGE-reranker cross-encoder."""
        if not passages:
            return []

        self._load_model("bge-reranker")
        self._reset_idle_timer("bge-reranker")

        if "bge-reranker" not in self._models:
            return self._heuristic_rerank(query, passages)

        model = self._models["bge-reranker"]
        tokenizer = self._tokenizers.get("bge-reranker")

        if tokenizer is None:
            return self._heuristic_rerank(query, passages)

        try:
            import torch

            scores = []
            batch_size = 8
            for i in range(0, len(passages), batch_size):
                batch = passages[i : i + batch_size]
                pairs = [[query, p] for p in batch]
                encoding = tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoding = {k: v.to(self._device) for k, v in encoding.items()}

                with torch.no_grad():
                    outputs = model(**encoding)
                    logits = outputs.logits.squeeze(-1)
                    batch_scores = logits.cpu().tolist()
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    scores.extend(batch_scores)

            return scores

        except Exception as e:
            logger.warning("BGE-reranker failed: %s; using heuristic rerank", e)
            return self._heuristic_rerank(query, passages)

    def _heuristic_rerank(self, query: str, passages: List[str]) -> List[float]:
        """Simple keyword overlap reranker fallback."""
        import re
        query_terms = set(re.findall(r"\w+", query.lower()))
        scores = []
        for passage in passages:
            passage_terms = set(re.findall(r"\w+", passage.lower()))
            if not query_terms:
                scores.append(0.0)
            else:
                overlap = len(query_terms & passage_terms) / len(query_terms)
                scores.append(overlap)
        return scores

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Abstractive summarization via BART-CNN."""
        if not text or len(text.split()) < 20:
            return text

        self._load_model("bart-cnn")
        self._reset_idle_timer("bart-cnn")

        if "bart-cnn" not in self._models:
            return self._extractive_summary_fallback(text, max_length)

        pipe = self._models["bart-cnn"]
        try:
            # BART max input length is 1024 tokens; truncate input text
            truncated = " ".join(text.split()[:800])
            result = pipe(
                truncated,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"]
        except Exception as e:
            logger.warning("BART summarization failed: %s; using extractive fallback", e)
            return self._extractive_summary_fallback(text, max_length)

    def _extractive_summary_fallback(self, text: str, max_length: int = 150) -> str:
        """Simple first-N-words extractive summary fallback."""
        words = text.split()
        word_limit = max_length * 2 // 3  # approximate token/word ratio
        if len(words) <= word_limit:
            return text
        return " ".join(words[:word_limit]) + "..."
