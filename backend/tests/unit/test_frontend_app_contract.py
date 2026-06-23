from __future__ import annotations

import json
import subprocess
from pathlib import Path


APP_JS = Path(__file__).parents[3] / "frontend" / "web" / "app.js"
OPTION_CHAIN_JS = Path(__file__).parents[3] / "frontend" / "web" / "OptionChain.js"


def _run_helper_expression(expression: str):
    script = f"""
const fs = require("fs");
const vm = require("vm");
const source = fs.readFileSync({json.dumps(str(APP_JS))}, "utf8");
const helperSource = source.slice(0, source.indexOf("function formatFlag"));
const context = {{
  React: {{}},
  ReactDOM: {{}},
  window: {{}},
  console,
}};
vm.createContext(context);
vm.runInContext(helperSource, context);
const result = vm.runInContext({json.dumps(expression)}, context);
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", script],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_diagnostic_helpers_traverse_and_count_candidate_diagnostics() -> None:
    result = _run_helper_expression(
        """(() => {
          const payload = {
            details: {
              option_selection: {
                candidate_diagnostics: [
                  { strike: 25000, rejections: ["stale quote"] },
                  { strike: 25100, rejections: ["low OI"] },
                ],
              },
            },
          };
          const rows = diagnosticValue(
            [payload],
            ["candidate_diagnostics", "candidates", "candidate_contracts", "contracts_checked"],
          );
          return {
            count: Array.isArray(rows) ? rows.length : null,
            reasons: diagnosticReasons(payload),
          };
        })()"""
    )

    assert result == {"count": 2, "reasons": ["stale quote", "low OI"]}


def test_null_quote_numbers_remain_missing() -> None:
    result = _run_helper_expression(
        """({
          nullNumber: nullableNumber(null),
          emptyNumber: nullableNumber(""),
          zeroNumber: nullableNumber(0),
          quote: quoteProvenance({ quote_source: "upstox", quote_age_seconds: null }),
        })"""
    )

    assert result["nullNumber"] is None
    assert result["emptyNumber"] is None
    assert result["zeroNumber"] == 0
    assert result["quote"]["ageSeconds"] is None


def test_pending_position_controls_and_successful_trade_wording_are_present() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert '["ENTRY_PENDING", "EXIT_SUBMITTING", "EXIT_PENDING"].includes(status)' in source
    assert source.count("disabled={reconciliationPending}") == 2
    assert "Entry pending reconciliation" in source
    assert "Exit pending reconciliation" in source
    assert 'label="Successful Trades / Symbol"' in source


def test_chart_ui_is_fixed_to_execution_aligned_one_minute_mode() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "Recent sessions / 1m" in source
    assert "changeChartRange" not in source
    assert "changeChartInterval" not in source
    assert "warmChartRange" not in source
    assert "loadMoreChartHistory" not in source
    assert "markerFromSignal" not in source
    assert "mergeSignalMarker" not in source
    assert 'range: rangeKey || "1d"' not in source
    assert 'interval: intervalKey || "1minute"' not in source


def test_execution_ui_exposes_sandbox_and_live_without_paper_mode() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'updateMode("sandbox")' in source
    assert 'updateMode("live")' in source
    assert "Sandbox Portfolio" in source
    assert "Upstox Sandbox Token" in source
    assert "/execution/sandbox/reset" in source
    assert "/execution/sandbox/test-order" in source
    assert 'updateMode("paper")' not in source
    assert "Paper Portfolio" not in source
    assert "/execution/paper/reset" not in source


def test_option_chain_auto_refreshes_live_quotes_without_browser_reload() -> None:
    source = OPTION_CHAIN_JS.read_text(encoding="utf-8")

    assert "const AUTO_REFRESH_MS = 1000;" in source
    assert "fetchChain(false, false)" in source
    assert "cache: 'no-store'" in source
    assert "document.visibilityState !== 'visible'" in source


def test_strike_chart_refreshes_from_stored_live_quotes() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "const STRIKE_CHART_REFRESH_MS = 4000;" in source
    assert 'limit: "2000"' in source
    assert 'refresh: "false"' in source
    assert "window.setInterval(refreshVisibleChart, STRIKE_CHART_REFRESH_MS)" in source
