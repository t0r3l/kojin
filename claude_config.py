from pathlib import Path
import os
from typing import Optional, Dict


def load_claude_key(path: Optional[str] = None) -> Optional[str]:
    """Load Claude API key.

    Search order:
    1. Environment variable `CLAUDE_API_KEY`
    2. Explicit `path` argument (file containing the key)
    3. A file named `.claude_key` in the current working directory
    4. A `.env` file loaded via `python-dotenv` if available

    Returns the key string (stripped) or None if not found.
    """
    key = os.getenv("CLAUDE_API_KEY")
    if key:
        return key.strip()

    if path:
        p = Path(path)
        if p.exists():
            return p.read_text().strip()

    p = Path(".claude_key")
    if p.exists():
        return p.read_text().strip()

    # Try loading .env via python-dotenv (optional)
    try:
        from dotenv import load_dotenv

        load_dotenv()
        key = os.getenv("CLAUDE_API_KEY")
        if key:
            return key.strip()
    except Exception:
        pass

    return None


def api_base_url() -> str:
    """Return the Claude base URL, defaulting to Anthropic's API."""
    return os.getenv("CLAUDE_API_URL", "https://api.anthropic.com")


def get_auth_headers() -> Dict[str, str]:
    """Return headers for requests. Raises RuntimeError if key not found."""
    key = load_claude_key()
    if not key:
        raise RuntimeError(
            "Claude API key not found. Set CLAUDE_API_KEY or create a .claude_key file."
        )
    return {"x-api-key": key}


def make_request_json(session_or_requests, path: str, json_body: dict, timeout: int = 30):
    """Convenience wrapper for posting JSON to the Claude API.

    `session_or_requests` may be the `requests` module or a `requests.Session()`.
    Returns the response object from that caller.
    """
    headers = get_auth_headers()
    url = api_base_url().rstrip("/") + "/" + path.lstrip("/")
    # session_or_requests should support post(url, json=..., headers=...)
    return session_or_requests.post(url, json=json_body, headers=headers, timeout=timeout)
