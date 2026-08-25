"""
Wraps `ollama pull` / `ollama rm` / `ollama list` for the pull -> generate
-> delete -> next teacher-data workflow (Phase 11). Deliberately never
deletes a model this process didn't pull itself -- see
scripts/generate_teacher_data.py for the safety logic that decides which
models are eligible for cleanup.
"""
import subprocess
import time
from typing import List


class OllamaManagerError(RuntimeError):
    pass


def list_models(ollama_bin: str = "ollama") -> List[str]:
    """Return the tags of every model currently present locally."""
    result = subprocess.run(
        [ollama_bin, "list"], capture_output=True, text=True, timeout=30,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise OllamaManagerError(f"`ollama list` failed: {result.stderr}")
    lines = result.stdout.strip().splitlines()[1:]  # skip header row
    return [line.split()[0] for line in lines if line.strip()]


def is_present(model: str, ollama_bin: str = "ollama") -> bool:
    return model in list_models(ollama_bin)


def pull_model(model: str, ollama_bin: str = "ollama", timeout: int = 1800, retries: int = 3) -> None:
    """`ollama pull` writes a redrawing progress bar to stdout via
    carriage returns -- captured through a pipe, every redraw becomes a
    separate chunk, so a multi-GB pull can produce hundreds of KB of
    useless text. We don't need that output, only success/failure, so
    stdout is discarded rather than captured. Ollama resumes a pull from
    its partial blob on retry (confirmed by the `-partial` chunk files
    it keeps under the models directory), so retrying after a transient
    network error picks up roughly where it left off rather than
    restarting the whole download."""
    last_err = None
    for attempt in range(retries):
        try:
            result = subprocess.run(
                [ollama_bin, "pull", model],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                timeout=timeout, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            # a stalled/very slow connection -- not a returncode failure, so
            # it has to be caught separately or it propagates straight out
            # of pull_model as a raw TimeoutExpired, uncaught anywhere
            # upstream, killing the whole run instead of just this attempt
            last_err = f"timed out after {timeout}s"
        else:
            if result.returncode == 0:
                return
            last_err = result.stderr
        if attempt < retries - 1:
            time.sleep(5 * (attempt + 1))
    raise OllamaManagerError(f"`ollama pull {model}` failed after {retries} attempts: {last_err}")


def delete_model(model: str, ollama_bin: str = "ollama") -> None:
    result = subprocess.run(
        [ollama_bin, "rm", model], capture_output=True, text=True, timeout=60,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise OllamaManagerError(f"`ollama rm {model}` failed: {result.stderr}")
