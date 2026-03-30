"""
Utility helpers used across the project.
"""

import re
import json
from datetime import datetime
from pathlib import Path


def clean_text(text: str) -> str:
    """Normalize whitespace and strip leading/trailing spaces."""
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_chars: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def format_ms(ms: int) -> str:
    """Format milliseconds into a human-readable string."""
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def score_color(score: float, max_val: float = 1.0) -> str:
    """Return a hex color for a score (green = good, red = bad)."""
    ratio = score / max_val
    if ratio >= 0.75:
        return "#34d399"  # green
    elif ratio >= 0.5:
        return "#f59e0b"  # amber
    else:
        return "#f87171"  # red


def export_results_json(results: list, output_path: str = None) -> str:
    """
    Export a list of EvalResult dicts to JSON.
    Returns the JSON string and optionally writes to file.
    """
    data = {
        "exported_at": datetime.now().isoformat(),
        "count": len(results),
        "results": results,
    }
    json_str = json.dumps(data, indent=2, default=str)
    if output_path:
        Path(output_path).write_text(json_str)
    return json_str


def load_sample_prompts() -> list[dict]:
    """Return sample prompts for demo/testing purposes."""
    return [
        {
            "name": "Direct Question",
            "text": "What is gradient descent? Explain briefly.",
        },
        {
            "name": "ELI5 Style",
            "text": "Explain gradient descent like I'm 10 years old, using a simple analogy.",
        },
        {
            "name": "Expert Framing",
            "text": (
                "You are an ML researcher. Explain gradient descent mathematically, "
                "including the update rule and convergence conditions."
            ),
        },
        {
            "name": "Step-by-Step",
            "text": (
                "Walk me through gradient descent step by step. "
                "Number each step and explain what happens at each one."
            ),
        },
        {
            "name": "With Examples",
            "text": (
                "Explain gradient descent and give a concrete numerical example "
                "showing how the weights update over two iterations."
            ),
        },
    ]
