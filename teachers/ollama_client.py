"""HTTP client for a local Ollama server's generate endpoint -- no API key,
just a plain POST to localhost. Uses urllib (stdlib), consistent with
scripts/download_corpus.py's approach elsewhere in this project."""
import json
import urllib.error
import urllib.request
from typing import Optional


class OllamaError(RuntimeError):
    pass


def generate(
    prompt: str,
    model: str,
    host: str = "http://localhost:11434",
    system: Optional[str] = None,
    temperature: float = 0.7,
    timeout: int = 120,
) -> str:
    """One-shot (non-streaming) completion from a local Ollama model.
    Raises OllamaError on any failure (connection refused, model not
    found, bad response shape) -- never returns a partial/garbage string
    silently."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system

    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(
            f"could not reach Ollama at {host} (is `ollama serve` running?): {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise OllamaError(f"Ollama returned non-JSON response: {e}") from e

    if "response" not in body:
        raise OllamaError(f"Ollama response missing 'response' field: {body}")
    return body["response"]
