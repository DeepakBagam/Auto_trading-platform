from __future__ import annotations

try:
    from _bootstrap import bootstrap_project_root
except Exception:
    from scripts._bootstrap import bootstrap_project_root

bootstrap_project_root()

import json
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import requests

from utils.config import get_settings


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "*" * len(value)
    return value[:3] + "*" * (len(value) - 6) + value[-3:]


def _set_env_key(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    found = False
    updated = []
    for line in lines:
        if line.startswith(f"{key}="):
            updated.append(f"{key}={value}")
            found = True
        else:
            updated.append(line)
    if not found:
        updated.append(f"{key}={value}")
    env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _extract_code(user_input: str) -> str:
    text = user_input.strip()
    if "http://" in text or "https://" in text:
        parsed = urlparse(text)
        query = parse_qs(parsed.query)
        return (query.get("code") or [""])[0]
    return text


def _ensure_env_exists() -> None:
    env_path = Path(".env")
    if env_path.exists():
        return
    example = Path(".env.example")
    if example.exists():
        env_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        env_path.write_text("", encoding="utf-8")


def main() -> None:
    _ensure_env_exists()
    settings = get_settings()

    api_key = settings.upstox_api_key
    api_secret = settings.upstox_api_secret
    redirect_uri = settings.upstox_redirect_uri

    if not api_key or not api_secret or not redirect_uri:
        raise RuntimeError(
            "Set UPSTOX_API_KEY, UPSTOX_API_SECRET, and UPSTOX_REDIRECT_URI in .env first."
        )

    auth_url = (
        "https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code&client_id={quote(api_key)}&redirect_uri={quote(redirect_uri)}"
    )
    print("\nStep 1: Open this URL in browser and login/approve:")
    print(auth_url)
    print("\nStep 2: Paste either the full redirected URL or just the `code` value:")
    code = _extract_code(input("> "))
    if not code:
        raise RuntimeError("Authorization code missing.")

    token_url = "https://api.upstox.com/v2/login/authorization/token"
    payload = {
        "code": code,
        "client_id": api_key,
        "client_secret": api_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded",
    }
    resp = requests.post(token_url, data=payload, headers=headers, timeout=30)
    if resp.status_code >= 400:
        raise RuntimeError(f"Token request failed [{resp.status_code}]: {resp.text}")

    data = resp.json()
    token = data.get("access_token") or (data.get("data") or {}).get("access_token")
    if not token:
        raise RuntimeError(f"Access token missing in response: {json.dumps(data)}")

    env_path = Path(".env")
    _set_env_key(env_path, "UPSTOX_ACCESS_TOKEN", token)
    print("\nAccess token saved to .env as UPSTOX_ACCESS_TOKEN.")
    print(f"Token preview: {_mask(token)}")


if __name__ == "__main__":
    main()
