"""
Evaluation Pipeline — Enhanced
Supports 200+ prompt variants with concurrent execution,
streaming progress callbacks, and per-variant analysis.
"""

import time
import uuid
import concurrent.futures
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

from src.models.ollama_connector import generate
from src.evaluator.metrics import (
    bleu_score,
    rouge_l_score,
    semantic_similarity,
    coherence_score,
    composite_score,
)
from src.evaluator.llm_judge import judge_response


@dataclass
class EvalResult:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt_id: str = ""
    prompt_name: str = ""
    prompt_text: str = ""
    task_description: str = ""
    reference_answer: str = ""
    llm_response: str = ""
    model: str = ""
    temperature: float = 0.7

    bleu: float = 0.0
    rouge_l: float = 0.0
    semantic_similarity: float = 0.0
    coherence: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    usefulness: float = 0.0
    judge_reasoning: str = ""
    judge_used: bool = False

    composite: float = 0.0

    latency_ms: int = 0
    timestamp: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def metric_summary(self) -> dict:
        return {
            "Composite": round(self.composite, 2),
            "Semantic": round(self.semantic_similarity, 4),
            "Coherence": round(self.coherence, 4),
            "Faithfulness": round(self.faithfulness, 4),
            "Completeness": round(self.completeness, 4),
            "Clarity": round(self.clarity, 4),
            "Usefulness": round(self.usefulness, 4),
            "BLEU": round(self.bleu, 4),
            "ROUGE-L": round(self.rouge_l, 4),
            "Latency (ms)": self.latency_ms,
        }


def run_evaluation(
    prompt_text: str,
    task_description: str,
    reference_answer: str = "",
    prompt_name: str = "Unnamed Prompt",
    prompt_id: str = "",
    model: str = "llama3",
    temperature: float = 0.7,
    use_llm_judge: bool = True,
) -> EvalResult:
    import datetime

    result = EvalResult(
        prompt_id=prompt_id or str(uuid.uuid4())[:8],
        prompt_name=prompt_name,
        prompt_text=prompt_text,
        task_description=task_description,
        reference_answer=reference_answer,
        model=model,
        temperature=temperature,
        timestamp=datetime.datetime.now().isoformat(),
    )

    try:
        start = time.time()
        result.llm_response = generate(
            prompt=prompt_text,
            model=model,
            temperature=temperature,
        )
        result.latency_ms = int((time.time() - start) * 1000)
    except Exception as e:
        result.error = str(e)
        return result

    if reference_answer.strip():
        result.bleu = bleu_score(result.llm_response, reference_answer)
        result.rouge_l = rouge_l_score(result.llm_response, reference_answer)
        result.semantic_similarity = semantic_similarity(
            result.llm_response, reference_answer
        )
    else:
        result.semantic_similarity = semantic_similarity(
            result.llm_response, task_description
        )
        result.bleu = 0.0
        result.rouge_l = 0.0

    result.coherence = coherence_score(result.llm_response)

    if use_llm_judge:
        judge = judge_response(
            task_prompt=task_description,
            response=result.llm_response,
            model=model,
        )
        result.faithfulness    = judge.get("faithfulness", 0.0)
        result.completeness    = judge.get("completeness", 0.0)
        result.clarity         = judge.get("clarity", 0.0)
        result.usefulness      = judge.get("usefulness", 0.0)
        result.judge_reasoning = judge.get("reasoning", "")
        result.judge_used      = judge.get("judge_used", False)
    else:
        result.faithfulness = result.semantic_similarity
        result.completeness = min(1.0, len(result.llm_response.split()) / 80)
        result.clarity      = result.coherence
        result.usefulness   = (result.faithfulness + result.clarity) / 2

    result.composite = composite_score(
        {
            "bleu": result.bleu if reference_answer else result.faithfulness,
            "rouge_l": result.rouge_l if reference_answer else result.completeness,
            "semantic_similarity": result.semantic_similarity,
            "coherence": result.coherence,
            "faithfulness": result.faithfulness,
        }
    )

    return result


def run_batch_evaluation(
    prompts: list,
    task_description: str,
    reference_answer: str = "",
    model: str = "llama3",
    temperature: float = 0.7,
    use_llm_judge: bool = True,
    max_workers: int = 1,
    progress_callback: Optional[Callable] = None,
    chunk_size: int = 50,
) -> list:
    """
    Evaluate 200+ prompt variants efficiently.

    Args:
        prompts: List of {"name": str, "text": str, "id": str} dicts
        max_workers: Parallel threads (1 = sequential, 2-4 = parallel)
        progress_callback: Called with (completed, total, latest_result) after each
        chunk_size: Process in chunks to avoid memory pressure with 200+ variants
    """
    results = []
    total = len(prompts)
    completed = 0

    def _eval_one(p):
        return run_evaluation(
            prompt_text=p["text"],
            task_description=task_description,
            reference_answer=reference_answer,
            prompt_name=p.get("name", "Prompt"),
            prompt_id=p.get("id", ""),
            model=model,
            temperature=temperature,
            use_llm_judge=use_llm_judge,
        )

    for chunk_start in range(0, total, chunk_size):
        chunk = prompts[chunk_start: chunk_start + chunk_size]

        if max_workers <= 1:
            for p in chunk:
                r = _eval_one(p)
                results.append(r)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, r)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_map = {ex.submit(_eval_one, p): p for p in chunk}
                for future in concurrent.futures.as_completed(future_map):
                    r = future.result()
                    results.append(r)
                    completed += 1
                    if progress_callback:
                        progress_callback(completed, total, r)

    results.sort(key=lambda r: r.composite, reverse=True)
    return results


def generate_variant_report(results: list) -> dict:
    """Per-variant analysis report with summary stats."""
    if not results:
        return {}

    scores = [r.composite for r in results if not r.error]
    metrics_keys = ["bleu", "rouge_l", "semantic_similarity",
                    "coherence", "faithfulness", "completeness",
                    "clarity", "usefulness"]

    per_variant = []
    for rank, r in enumerate(results, 1):
        entry = {
            "rank": rank,
            "name": r.prompt_name,
            "composite": round(r.composite, 2),
            "metrics": {k: round(getattr(r, k, 0), 4) for k in metrics_keys},
            "latency_ms": r.latency_ms,
            "response_length": len(r.llm_response.split()) if r.llm_response else 0,
            "judge_used": r.judge_used,
            "judge_reasoning": r.judge_reasoning,
            "error": r.error,
            "response_preview": (r.llm_response or "")[:300],
        }
        per_variant.append(entry)

    best  = results[0]
    worst = results[-1] if len(results) > 1 else results[0]

    import statistics
    discriminating = None
    max_std = 0.0
    for k in metrics_keys:
        vals = [getattr(r, k, 0) for r in results if not r.error]
        if len(vals) >= 2:
            s = statistics.stdev(vals)
            if s > max_std:
                max_std = s
                discriminating = k

    return {
        "total_variants": len(results),
        "successful": len([r for r in results if not r.error]),
        "failed": len([r for r in results if r.error]),
        "score_range": {
            "min": round(min(scores), 2) if scores else 0,
            "max": round(max(scores), 2) if scores else 0,
            "mean": round(sum(scores) / len(scores), 2) if scores else 0,
            "spread": round(max(scores) - min(scores), 2) if scores else 0,
        },
        "best_variant": {
            "name": best.prompt_name,
            "score": round(best.composite, 2),
            "key_strengths": [k for k in metrics_keys if getattr(best, k, 0) >= 0.7],
        },
        "worst_variant": {"name": worst.prompt_name, "score": round(worst.composite, 2)},
        "most_discriminating_metric": discriminating,
        "per_variant": per_variant,
    }
