from backend.api.main import _should_rate_limit_execution_request


def test_execution_mode_and_read_endpoints_are_not_rate_limited() -> None:
    assert _should_rate_limit_execution_request("POST", "/execution/mode") is False
    assert _should_rate_limit_execution_request("GET", "/execution/status") is False
    assert _should_rate_limit_execution_request("GET", "/execution/portfolio") is False


def test_high_risk_execution_mutations_are_rate_limited() -> None:
    assert _should_rate_limit_execution_request("POST", "/execution/run-once") is True
    assert _should_rate_limit_execution_request("POST", "/execution/emergency-exit") is True
    assert _should_rate_limit_execution_request("POST", "/execution/positions/12/close") is True
    assert _should_rate_limit_execution_request("DELETE", "/execution/positions/12") is True
