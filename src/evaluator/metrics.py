"""
Evaluation Metrics
Computes BLEU, ROUGE-L, and Semantic Similarity scores.
All functions take (hypothesis, reference) string pairs and return float 0-1.
"""

import re
import math
from collections import Counter
from typing import Optional

# ── BLEU ──────────────────────────────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """
    Compute corpus-level BLEU score (1-4 gram) with brevity penalty.
    Returns float in [0, 1].
    """
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hyp_tokens, n)
        ref_ng = _ngrams(ref_tokens, n)
        clipped = sum(min(count, ref_ng[gram]) for gram, count in hyp_ng.items())
        total = max(len(hyp_tokens) - n + 1, 0)
        precisions.append(clipped / total if total > 0 else 0.0)

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    return round(bp * math.exp(log_avg), 4)


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def _lcs_length(a: list, b: list) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 score.
    Returns float in [0, 1].
    """
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    lcs = _lcs_length(hyp, ref)
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ── Semantic Similarity ───────────────────────────────────────────────────────

_model = None


def _get_embedding_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _model


def semantic_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between sentence embeddings.
    Returns float in [0, 1]. Higher = more similar meaning.
    """
    try:
        import numpy as np
        model = _get_embedding_model()
        embs = model.encode([text_a, text_b], normalize_embeddings=True)
        score = float(np.dot(embs[0], embs[1]))
        return round(max(0.0, min(1.0, score)), 4)
    except Exception as e:
        # Fallback: simple word overlap jaccard
        a_words = set(text_a.lower().split())
        b_words = set(text_b.lower().split())
        if not a_words or not b_words:
            return 0.0
        return round(len(a_words & b_words) / len(a_words | b_words), 4)


# ── Coherence (heuristic) ─────────────────────────────────────────────────────

def coherence_score(text: str) -> float:
    """
    Heuristic coherence score based on:
    - Sentence count (not too short, not too long)
    - Avg sentence length
    - Presence of transition words
    - No excessive repetition
    Returns float in [0, 1].
    """
    if not text.strip():
        return 0.0

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    n_sentences = len(sentences)

    # Penalise very short or very long responses
    length_score = min(1.0, n_sentences / 5) if n_sentences < 5 else min(1.0, 10 / n_sentences)

    # Avg word count per sentence
    avg_words = sum(len(s.split()) for s in sentences) / max(n_sentences, 1)
    word_score = min(1.0, avg_words / 20)

    # Transition word bonus
    transitions = {
        "however", "therefore", "furthermore", "additionally", "consequently",
        "first", "second", "finally", "in conclusion", "for example",
        "in addition", "on the other hand", "as a result", "in summary",
    }
    text_lower = text.lower()
    transition_bonus = min(0.2, sum(0.04 for t in transitions if t in text_lower))

    # Repetition penalty
    words = text.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    repetition_penalty = 0.0 if unique_ratio > 0.5 else (0.5 - unique_ratio)

    score = (length_score * 0.3 + word_score * 0.4 + transition_bonus + 0.3) - repetition_penalty
    return round(max(0.0, min(1.0, score)), 4)


# ── Composite Score ───────────────────────────────────────────────────────────

WEIGHTS = {
    "bleu": 0.15,
    "rouge_l": 0.15,
    "semantic_similarity": 0.40,
    "coherence": 0.15,
    "faithfulness": 0.15,
}


def composite_score(scores: dict) -> float:
    """
    Weighted composite of all metrics.
    Returns float in [0, 100].
    """
    total = sum(scores.get(k, 0) * w for k, w in WEIGHTS.items())
    return round(total * 100, 2)
