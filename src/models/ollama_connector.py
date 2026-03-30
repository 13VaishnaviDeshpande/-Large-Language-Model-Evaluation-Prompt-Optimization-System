"""
Ollama Connector
Wraps the local Ollama HTTP API.
"""

import requests

OLLAMA_BASE = "http://localhost:11434"


def is_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_models() -> list:
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def generate(
    prompt: str,
    model: str = "llama3",
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    if system:
        payload["system"] = system

    r = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json=payload,
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "")
