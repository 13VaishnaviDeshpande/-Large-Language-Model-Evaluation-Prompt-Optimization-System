"""
LLM-as-Judge
Uses a second LLM call to score response quality and faithfulness.
This is the most powerful evaluation method for open-ended tasks.
"""

import re
import json
from src.models.ollama_connector import generate


JUDGE_SYSTEM_PROMPT = """You are a strict, impartial evaluator of AI-generated text.
You will be given a task prompt and a response. Score the response on the following criteria.
Respond ONLY with a valid JSON object. Do not add any explanation outside the JSON."""


JUDGE_PROMPT_TEMPLATE = """Task Prompt:
{task_prompt}

Response to Evaluate:
{response}

Score this response on each criterion from 0.0 to 1.0:
- faithfulness: Does the response directly address what was asked? Does it stay on topic without hallucinating?
- completeness: Does it cover all aspects of the task?
- clarity: Is it clear, well-structured, and easy to understand?
- usefulness: Would a real user find this genuinely helpful?

Respond with ONLY this JSON (no markdown, no preamble):
{{
  "faithfulness": <float>,
  "completeness": <float>,
  "clarity": <float>,
  "usefulness": <float>,
  "reasoning": "<1-2 sentence explanation>"
}}"""


def judge_response(
    task_prompt: str,
    response: str,
    model: str = "llama3",
) -> dict:
    """
    Use an LLM to evaluate a response against a task prompt.
    Returns dict with scores and reasoning.
    Falls back to heuristics if LLM judge fails.
    """
    judge_input = JUDGE_PROMPT_TEMPLATE.format(
        task_prompt=task_prompt,
        response=response,
    )

    try:
        raw = generate(
            prompt=judge_input,
            model=model,
            system=JUDGE_SYSTEM_PROMPT,
            temperature=0.1,  # Low temp for consistent scoring
            max_tokens=300,
        )
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Clamp all numeric values to [0, 1]
            for key in ["faithfulness", "completeness", "clarity", "usefulness"]:
                if key in result:
                    result[key] = max(0.0, min(1.0, float(result[key])))
            result["judge_used"] = True
            return result
    except Exception as e:
        pass  # Fall through to heuristic fallback

    # Heuristic fallback when LLM judge fails
    return _heuristic_judge(task_prompt, response)


def _heuristic_judge(task_prompt: str, response: str) -> dict:
    """
    Simple heuristic fallback when LLM judging is unavailable.
    Not as accurate but ensures the pipeline never breaks.
    """
    from src.evaluator.metrics import semantic_similarity, coherence_score

    faithfulness = semantic_similarity(response, task_prompt)
    coherence = coherence_score(response)
    word_count = len(response.split())
    completeness = min(1.0, word_count / 100)  # Rough proxy

    return {
        "faithfulness": faithfulness,
        "completeness": completeness,
        "clarity": coherence,
        "usefulness": (faithfulness + coherence) / 2,
        "reasoning": "Heuristic scoring (LLM judge unavailable)",
        "judge_used": False,
    }
