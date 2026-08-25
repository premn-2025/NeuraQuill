"""
A minimal .env loader -- no python-dotenv dependency, consistent with this
project's preference for hand-rolled infrastructure over new dependencies
where the need is this small. Parses simple KEY=VALUE lines, sets each into
os.environ only if not already set there (so real environment variables
always take precedence over the file, matching standard .env conventions).
"""
import os
from pathlib import Path
from typing import Dict, Optional


def parse_env_file(text: str) -> Dict[str, str]:
    """Parse KEY=VALUE lines. Blank lines and lines starting with '#' are
    ignored. Values may be wrapped in matching single or double quotes,
    which are stripped."""
    result: Dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            result[key] = value
    return result


def load_env(path: str = ".env") -> Dict[str, str]:
    """Load `path` (if it exists) and set each KEY=VALUE into os.environ,
    without overriding a variable that's already set in the real
    environment. Returns the parsed dict regardless of whether the file
    existed (empty dict if not)."""
    file_path = Path(path)
    if not file_path.exists():
        return {}
    parsed = parse_env_file(file_path.read_text(encoding="utf-8"))
    for key, value in parsed.items():
        os.environ.setdefault(key, value)
    return parsed


def require_env(name: str) -> str:
    """Get an environment variable, raising a clear error (not a bare
    KeyError) if it's unset or empty -- e.g. an .env with a blank
    ANTHROPIC_API_KEY= line."""
    value = os.environ.get(name, "")
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            f"or export {name} in your shell."
        )
    return value


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, "")
    return value if value else default
