from backend.api.routes.execution import _check_upstox_profile
from backend.utils.config import Settings


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def test_check_upstox_profile_reports_ok(monkeypatch) -> None:
    captured = {}

    def fake_get(url, *, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Response(
            200,
            {"data": {"user_name": "Trader", "email": "trader@example.test", "broker": "upstox"}},
        )

    monkeypatch.setattr("backend.api.routes.execution._requests.get", fake_get)

    out = _check_upstox_profile(Settings(_env_file=None, upstox_base_url="https://example.test"), "token")

    assert out["status"] == "ok"
    assert out["user_name"] == "Trader"
    assert captured["url"] == "https://example.test/v2/user/profile"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["timeout"] == 5


def test_check_upstox_profile_reports_expired_token(monkeypatch) -> None:
    monkeypatch.setattr(
        "backend.api.routes.execution._requests.get",
        lambda *args, **kwargs: _Response(401),
    )

    out = _check_upstox_profile(Settings(_env_file=None), "expired-token")

    assert out["status"] == "error"
    assert "expired" in out["detail"]
