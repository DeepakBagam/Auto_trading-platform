const state = {
  chart: null,
  candleSeries: null,
  symbol: "",
  symbolSnapshots: {},
  calendarDate: (() => {
    const parts = new Intl.DateTimeFormat("en-IN", {
      timeZone: "Asia/Kolkata",
      year: "numeric",
      month: "2-digit"
    }).formatToParts(new Date());
    const year = Number(parts.find((part) => part.type === "year")?.value || new Date().getFullYear());
    const month = Number(parts.find((part) => part.type === "month")?.value || (new Date().getMonth() + 1));
    return { year, month };
  })(),
  latestConsensus: null,
  latestOptionSignal: null,
  requestSeq: 0,
  globalSnapshot: null,
  globalSnapshotAt: 0
};

const navButtons = [...document.querySelectorAll(".nav-link")];
const views = [...document.querySelectorAll(".view")];
const symbolSelect = document.getElementById("symbolSelect");
const apiStatus = document.getElementById("apiStatus");
const headerPrice = document.getElementById("headerPrice");
const headerSubtext = document.getElementById("headerSubtext");
const clockIst = document.getElementById("clockIst");
const emergencyStopBtn = document.getElementById("emergencyStopBtn");

const statWinRate = document.getElementById("statWinRate");
const statWinMeta = document.getElementById("statWinMeta");
const statPnl = document.getElementById("statPnl");
const statPnlMeta = document.getElementById("statPnlMeta");
const statOpenPositions = document.getElementById("statOpenPositions");
const statOpenMeta = document.getElementById("statOpenMeta");
const statDrawdown = document.getElementById("statDrawdown");
const statDrawdownMeta = document.getElementById("statDrawdownMeta");

const chartMeta = document.getElementById("chartMeta");
const mlDirection = document.getElementById("mlDirection");
const mlConfidenceBar = document.getElementById("mlConfidenceBar");
const mlConfidenceText = document.getElementById("mlConfidenceText");
const mlMoveText = document.getElementById("mlMoveText");
const mlStatusBadge = document.getElementById("mlStatusBadge");
const mlReason = document.getElementById("mlReason");
const pineDirection = document.getElementById("pineDirection");
const pineAgeText = document.getElementById("pineAgeText");
const pineFreshnessText = document.getElementById("pineFreshnessText");
const pineStatusBadge = document.getElementById("pineStatusBadge");
const pineReason = document.getElementById("pineReason");
const aiScoreText = document.getElementById("aiScoreText");
const aiGauge = document.getElementById("aiGauge");
const aiGaugeValue = document.getElementById("aiGaugeValue");
const aiStatusBadge = document.getElementById("aiStatusBadge");
const aiReason = document.getElementById("aiReason");
const verdictText = document.getElementById("verdictText");
const verdictMeta = document.getElementById("verdictMeta");

const openPositionsTable = document.getElementById("openPositionsTable");
const signalLogFeed = document.getElementById("signalLogFeed");
const optionDirection = document.getElementById("optionDirection");
const optionContract = document.getElementById("optionContract");
const optionExpiry = document.getElementById("optionExpiry");
const optionEntry = document.getElementById("optionEntry");
const optionRiskPlan = document.getElementById("optionRiskPlan");
const optionStrategy = document.getElementById("optionStrategy");
const optionReason = document.getElementById("optionReason");
const positionCards = document.getElementById("positionCards");
const mlReasons = document.getElementById("mlReasons");
const aiReasons = document.getElementById("aiReasons");
const consensusDetails = document.getElementById("consensusDetails");
const historyTable = document.getElementById("historyTable");
const ordersTable = document.getElementById("ordersTable");

const prevMonthBtn = document.getElementById("prevMonthBtn");
const nextMonthBtn = document.getElementById("nextMonthBtn");
const calendarMonthLabel = document.getElementById("calendarMonthLabel");
const calendarSummary = document.getElementById("calendarSummary");
const calendarGrid = document.getElementById("calendarGrid");
const calendarDayPanel = document.getElementById("calendarDayPanel");

const tradeModal = document.getElementById("tradeModal");
const modalTitle = document.getElementById("modalTitle");
const modalMeta = document.getElementById("modalMeta");
const miniChart = document.getElementById("miniChart");

function formatMoney(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : "-";
}

function formatSignedMoney(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    return "-";
  }
  const prefix = n >= 0 ? "+" : "";
  return `${prefix}${n.toFixed(2)}`;
}

function formatPct(value) {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(2)}%` : "-";
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? "-" : d.toLocaleString("en-IN", { hour12: false, timeZone: "Asia/Kolkata" });
}

function parseIstIsoToChartTime(value) {
  if (!value) {
    return null;
  }
  const text = String(value);
  const match = text.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/);
  if (match) {
    const [, year, month, day, hour, minute, second = "00"] = match;
    return Math.floor(Date.UTC(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second)
    ) / 1000);
  }
  const fallback = new Date(text);
  return Number.isNaN(fallback.getTime()) ? null : Math.floor(fallback.getTime() / 1000);
}

function formatIstTime(value = new Date()) {
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  }).format(value);
}

function normalizeSymbol(value) {
  const raw = String(value || "").replace(/[^a-z0-9]/gi, "").toUpperCase();
  if (raw === "BANKNIFTY" || raw === "NIFTYBANK") {
    return "BANKNIFTY";
  }
  return raw;
}

function sameSymbol(left, right) {
  return normalizeSymbol(left) === normalizeSymbol(right);
}

function formatAge(seconds) {
  const n = Number(seconds);
  if (!Number.isFinite(n)) {
    return "-";
  }
  const mins = Math.floor(n / 60);
  const secs = Math.floor(n % 60);
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
}

function setStatusBadge(element, kind, text) {
  element.className = "status-badge";
  element.classList.add(kind);
  element.textContent = text;
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: { "Cache-Control": "no-cache" },
    ...options
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function settledValue(result, fallback) {
  return result?.status === "fulfilled" ? result.value : fallback;
}

function settledError(result, label) {
  return result?.status === "rejected" ? `${label} offline` : null;
}

function defaultStats() {
  return {
    win_rate: 0,
    wins_today: 0,
    total_trades_today: 0,
    total_pnl_today: 0,
    open_positions_count: 0,
    open_positions_unrealized_pnl: 0,
    max_drawdown_percent: 0,
    profit_factor: 0
  };
}

function defaultGlobalSnapshot() {
  return {
    stats: defaultStats(),
    positions: [],
    history: [],
    _errors: []
  };
}

function defaultConsensus(symbol) {
  return {
    symbol,
    consensus: "HOLD",
    ml_signal: "NEUTRAL",
    ml_confidence: 0,
    ml_expected_move: 0,
    ml_reasons: [],
    pine_signal: "NEUTRAL",
    pine_age_seconds: null,
    ai_score: 0,
    ai_reasons: [],
    news_sentiment: 0,
    combined_score: 0,
    skip_reason: "Waiting for fresh live consensus",
    details: {}
  };
}

function defaultChartPayload(symbol) {
  return {
    symbol,
    actual: [],
    overlays: {},
    markers: [],
    freshness: {}
  };
}

function defaultOptionPayload() {
  return {
    option_signal: null,
    chain: [],
    expiry_date: null,
    strike_step: null
  };
}

function defaultSymbolSnapshot(symbol) {
  return {
    chart: defaultChartPayload(symbol),
    consensus: defaultConsensus(symbol),
    logs: [],
    optionPayload: defaultOptionPayload(),
    orders: [],
    _errors: []
  };
}

function activateView(name) {
  navButtons.forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  views.forEach((view) => view.classList.toggle("active", view.dataset.panel === name));
  if (name === "calendar") {
    loadCalendar();
  }
}

function updateClock() {
  clockIst.textContent = `IST ${formatIstTime(new Date())}`;
}

function createChart() {
  const chartEl = document.getElementById("chart");
  state.chart = LightweightCharts.createChart(chartEl, {
    width: chartEl.clientWidth,
    height: chartEl.clientHeight,
    layout: { background: { color: "#161b22" }, textColor: "#e6edf3" },
    rightPriceScale: { borderColor: "#30363d" },
    timeScale: { borderColor: "#30363d", timeVisible: true, secondsVisible: false },
    grid: {
      vertLines: { color: "rgba(255,255,255,0.04)" },
      horzLines: { color: "rgba(255,255,255,0.04)" }
    },
    localization: { locale: "en-IN" }
  });
  state.candleSeries = state.chart.addCandlestickSeries({
    upColor: "#00c853",
    downColor: "#ff3d00",
    wickUpColor: "#00c853",
    wickDownColor: "#ff3d00",
    borderVisible: false
  });
  window.addEventListener("resize", () => {
    if (state.chart) {
      state.chart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight });
    }
  });
}

function renderChart(payload) {
  const candles = (payload.actual || []).map((row) => ({
    time: parseIstIsoToChartTime(row.x),
    open: Number(row.open),
    high: Number(row.high),
    low: Number(row.low),
    close: Number(row.close)
  })).filter((row) => Number.isFinite(row.time));
  state.candleSeries.setData(candles);
  const markers = (payload.markers || []).map((row) => ({
    time: parseIstIsoToChartTime(row.time),
    position: row.position || "inBar",
    color: row.color,
    shape: row.shape || "circle",
    text: row.text || ""
  })).filter((row) => Number.isFinite(row.time));
  if (typeof state.candleSeries.setMarkers === "function") {
    state.candleSeries.setMarkers(markers);
  }
  if (candles.length > 0) {
    const last = candles[candles.length - 1];
    const prev = candles[Math.max(0, candles.length - 2)] || last;
    const change = last.close - prev.close;
    const changePct = prev.close ? (change / prev.close) * 100 : 0;
    headerPrice.textContent = `${state.symbol}: ${formatMoney(last.close)} (${change >= 0 ? "+" : ""}${changePct.toFixed(2)}%)`;
    headerPrice.style.color = change >= 0 ? "#00c853" : "#ff3d00";
  }
  const freshness = payload.freshness || {};
  const marketStatus = String(freshness.market_status || (freshness.is_live ? "live" : "stale")).replaceAll("_", " ");
  const latestSession = freshness.latest_session_date ? ` · session ${freshness.latest_session_date}` : "";
  chartMeta.textContent = `${state.symbol} · 1m candles · Pine 1m+3m+5m signals + trades · ${marketStatus}${latestSession}`;
}

function renderStats(stats) {
  statWinRate.textContent = formatPct(stats.win_rate);
  statWinMeta.textContent = `${stats.wins_today || 0} wins / ${stats.total_trades_today || 0} trades today`;
  statPnl.textContent = formatSignedMoney(stats.total_pnl_today);
  statPnl.className = Number(stats.total_pnl_today) >= 0 ? "pnl-profit" : "pnl-loss";
  statPnlMeta.textContent = "Realized intraday P&L";
  statOpenPositions.textContent = String(stats.open_positions_count || 0);
  statOpenMeta.textContent = `${formatSignedMoney(stats.open_positions_unrealized_pnl || 0)} unrealized`;
  statDrawdown.textContent = formatPct(stats.max_drawdown_percent);
  statDrawdownMeta.textContent = `Profit factor ${formatMoney(stats.profit_factor)}`;
}

function renderConsensus(consensus, optionPayload, openPositions) {
  state.latestConsensus = consensus;
  const pineMaxAge = Number(consensus?.details?.pine_max_age_seconds || 60);
  const mlPass = consensus.ml_signal === "BUY" || consensus.ml_signal === "SELL";
  mlDirection.textContent = consensus.ml_signal;
  mlConfidenceBar.style.width = `${Math.max(0, Math.min(100, Number(consensus.ml_confidence || 0) * 100))}%`;
  mlConfidenceText.textContent = `Confidence ${formatPct(Number(consensus.ml_confidence || 0) * 100)}`;
  mlMoveText.textContent = `Move ${formatMoney(consensus.ml_expected_move)} pts`;
  setStatusBadge(mlStatusBadge, mlPass ? "status-pass" : "status-wait", mlPass ? "PASS" : "WAIT");
  mlReason.textContent = consensus.ml_reasons?.[0] || consensus.skip_reason || "-";

  const pineFresh = Number(consensus.pine_age_seconds) < pineMaxAge && consensus.pine_signal === consensus.ml_signal;
  pineDirection.textContent = consensus.pine_signal || "NEUTRAL";
  pineAgeText.textContent = `Age ${formatAge(consensus.pine_age_seconds)}`;
  pineFreshnessText.textContent = pineFresh ? `Fresh <= ${pineMaxAge}s` : `Expired > ${pineMaxAge}s / mismatch`;
  setStatusBadge(
    pineStatusBadge,
    pineFresh ? "status-pass" : (mlPass ? "status-fail" : "status-wait"),
    pineFresh ? "PASS" : (mlPass ? "FAIL" : "WAIT")
  );
  pineReason.textContent = pineFresh ? "Pine matches ML direction" : (consensus.skip_reason || "Pine not aligned");

  const aiPass = Number(consensus.ai_score) >= 65;
  aiScoreText.textContent = `${formatMoney(consensus.ai_score)}`;
  aiGaugeValue.textContent = Math.round(Number(consensus.ai_score || 0));
  aiGauge.style.setProperty("--gauge-angle", `${Math.max(0, Math.min(360, Number(consensus.ai_score || 0) * 3.6))}deg`);
  setStatusBadge(
    aiStatusBadge,
    aiPass ? "status-pass" : (mlPass ? "status-fail" : "status-wait"),
    aiPass ? "PASS" : (mlPass ? "FAIL" : "WAIT")
  );
  aiReason.textContent = consensus.ai_reasons?.[0] || "-";

  const existing = openPositions[0];
  const strikeText = existing
    ? `${existing.strike} ${existing.option_type}`
    : optionPayload?.option_signal?.strike
      ? `${optionPayload.option_signal.strike} ${optionPayload.option_signal.option_type || ""}`
      : "-";
  if (consensus.consensus === "BUY" || consensus.consensus === "SELL") {
    verdictText.textContent = "TRADE SIGNAL";
    verdictMeta.textContent = `${consensus.consensus === "BUY" ? "BUY CE" : "BUY PE"} · ${strikeText}`;
  } else {
    verdictText.textContent = "NO TRADE";
    verdictMeta.textContent = consensus.skip_reason || "Waiting for all gates to align";
  }
}

function renderSignalLog(logs) {
  signalLogFeed.innerHTML = "";
  const rows = logs.slice(0, 10);
  if (!rows.length) {
    signalLogFeed.innerHTML = '<div class="empty-state">No signal logs yet.</div>';
    return;
  }
  rows.forEach((row) => {
    const item = document.createElement("div");
    item.className = "feed-item";
    item.innerHTML = `
      <div class="signal-row">
        <strong>${row.consensus}</strong>
        <span>${formatDateTime(row.timestamp)}</span>
      </div>
      <div class="tag-row">
        <span class="tag">${row.trade_placed ? "TRADE TAKEN" : "SKIPPED"}</span>
        <span class="tag">${row.ml_signal} / ${row.pine_signal}</span>
        <span class="tag">AI ${formatMoney(row.ai_score)}</span>
      </div>
      <div class="selection-reason">${row.skip_reason || "Consensus passed"}</div>
    `;
    signalLogFeed.appendChild(item);
  });
}

function createCloseButton(positionId) {
  return `<button class="close-button" data-close-position="${positionId}">Exit</button>`;
}

function renderOpenPositions(positions) {
  openPositionsTable.innerHTML = "";
  positionCards.innerHTML = "";
  if (!positions.length) {
    openPositionsTable.innerHTML = '<tr><td colspan="10" class="empty-state">No open positions.</td></tr>';
    positionCards.innerHTML = '<div class="empty-state">No open positions.</div>';
    return;
  }
  positions.forEach((row) => {
    const pnlClass = Number(row.unrealized_pnl) >= 0 ? "pnl-profit" : "pnl-loss";
    const tableRow = document.createElement("tr");
    tableRow.innerHTML = `
      <td>${row.symbol}</td>
      <td>${row.strike}</td>
      <td>${row.option_type}</td>
      <td>${formatMoney(row.entry_premium)}</td>
      <td>${formatMoney(row.current_premium)}</td>
      <td class="${pnlClass}">${formatSignedMoney(row.unrealized_pnl)}</td>
      <td>${formatMoney(row.current_sl)}</td>
      <td>${row.tsl_active ? "YES" : "NO"}</td>
      <td>${formatDateTime(row.entry_time)}</td>
      <td>${createCloseButton(row.position_id)}</td>
    `;
    openPositionsTable.appendChild(tableRow);

    const progressBase = Number(row.target_premium) - Number(row.entry_premium);
    const progress = progressBase > 0 ? Math.max(0, Math.min(100, ((Number(row.current_premium) - Number(row.entry_premium)) / progressBase) * 100)) : 0;
    const card = document.createElement("article");
    card.className = "position-card";
    card.innerHTML = `
      <div class="position-top">
        <div>
          <strong>${row.symbol} ${row.strike} ${row.option_type}</strong>
          <div class="selection-reason">Qty ${row.quantity} · Expiry ${row.expiry}</div>
        </div>
        ${createCloseButton(row.position_id)}
      </div>
      <div class="selection-row"><span>Entry Price</span><strong>${formatMoney(row.entry_premium)}</strong></div>
      <div class="selection-row"><span>Current Price</span><strong>${formatMoney(row.current_premium)}</strong></div>
      <div class="selection-row"><span>Unrealized P&amp;L</span><strong class="${pnlClass}">${formatSignedMoney(row.unrealized_pnl)}</strong></div>
      <div class="tag-row">
        <span class="tag">SL ${formatMoney(row.initial_sl)}</span>
        <span class="tag">${row.tsl_active ? `TSL ${formatMoney(row.current_sl)}` : `Current SL ${formatMoney(row.current_sl)}`}</span>
        ${row.tsl_active ? '<span class="tag warn">TSL ACTIVE</span>' : ""}
      </div>
      <div class="selection-row"><span>Target</span><strong>${formatMoney(row.target_premium)}</strong></div>
      <div class="progress"><div style="width:${progress}%"></div></div>
      <div class="selection-row"><span>Entry Time</span><strong>${formatDateTime(row.entry_time)}</strong></div>
    `;
    positionCards.appendChild(card);
  });
}

function renderOptionsDesk(optionPayload, positions, consensus) {
  const openPosition = positions[0];
  if (openPosition) {
    optionDirection.textContent = openPosition.option_type === "CE" ? "BUY CE" : "BUY PE";
    optionContract.textContent = `${openPosition.symbol} ${openPosition.strike} ${openPosition.option_type}`;
    optionExpiry.textContent = openPosition.expiry;
    optionEntry.textContent = formatMoney(openPosition.entry_premium);
    optionRiskPlan.textContent = `SL ${formatMoney(openPosition.current_sl)} / Target ${formatMoney(openPosition.target_premium)}`;
    optionStrategy.textContent = openPosition.strategy_name || "Consensus CE/PE";
    optionReason.textContent = openPosition.consensus_reason || "Live position is active.";
    return;
  }
  optionDirection.textContent = consensus?.consensus === "SELL" ? "BUY PE" : (consensus?.consensus === "BUY" ? "BUY CE" : "NO TRADE");
  optionContract.textContent = optionPayload?.option_signal?.strike ? `${state.symbol} ${optionPayload.option_signal.strike} ${optionPayload.option_signal.option_type || ""}` : "-";
  optionExpiry.textContent = optionPayload?.expiry_date || "-";
  optionEntry.textContent = formatMoney(optionPayload?.option_signal?.entry_price);
  optionRiskPlan.textContent = `SL ${formatMoney(optionPayload?.option_signal?.stop_loss)} / Target ${formatMoney(optionPayload?.option_signal?.take_profit)}`;
  optionStrategy.textContent = optionPayload?.option_signal?.strategy || "Consensus CE/PE";
  optionReason.textContent = optionPayload?.option_signal?.reasons?.join(" | ") || (consensus?.skip_reason || "Waiting for a tradeable setup.");
}

function renderList(container, values) {
  container.innerHTML = "";
  if (!values || !values.length) {
    container.innerHTML = '<div class="list-item">No details available.</div>';
    return;
  }
  values.forEach((value) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.textContent = value;
    container.appendChild(item);
  });
}

function renderConsensusDetails(consensus) {
  const details = consensus?.details || {};
  const entries = [
    ["ML Confidence", formatPct(Number(consensus?.ml_confidence || 0) * 100)],
    ["Expected Move", `${formatMoney(consensus?.ml_expected_move)} pts`],
    ["Pine Age", formatAge(consensus?.pine_age_seconds)],
    ["3m Bias", details.mtf_3m_action || "-"],
    ["5m Bias", details.mtf_5m_action || "-"],
    ["Regime", details.market_regime || "-"],
    ["AI Score", formatMoney(consensus?.ai_score)],
    ["Combined Score", formatMoney(consensus?.combined_score)],
    ["News Sentiment", formatMoney(consensus?.news_sentiment)],
    ["VIX Level", formatMoney(details.vix_level)],
    ["Latest Price", formatMoney(details.latest_price)],
    ["Predicted Close", formatMoney(details.predicted_close)]
  ];
  consensusDetails.innerHTML = entries.map(([label, value]) => `
    <div class="detail-item">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");
}

function renderHistory(history) {
  historyTable.innerHTML = "";
  if (!history.length) {
    historyTable.innerHTML = '<tr><td colspan="9" class="empty-state">No closed trades yet.</td></tr>';
    return;
  }
  history.forEach((row) => {
    const tr = document.createElement("tr");
    tr.dataset.trade = JSON.stringify(row);
    tr.innerHTML = `
      <td>${formatDateTime(row.entry_time)}</td>
      <td>${row.symbol}</td>
      <td>${row.strike}</td>
      <td>${row.option_type}</td>
      <td>${formatMoney(row.entry_premium)}</td>
      <td>${formatMoney(row.exit_premium)}</td>
      <td>${row.quantity}</td>
      <td class="${Number(row.realized_pnl) >= 0 ? "pnl-profit" : "pnl-loss"}">${formatSignedMoney(row.realized_pnl)}</td>
      <td>${row.exit_reason || "-"}</td>
    `;
    historyTable.appendChild(tr);
  });
}

function renderOrders(orders) {
  ordersTable.innerHTML = "";
  if (!orders.length) {
    ordersTable.innerHTML = '<tr><td colspan="12" class="empty-state">No recent orders for this symbol.</td></tr>';
    return;
  }
  orders.forEach((row) => {
    const realizedPnl = Number(row.realized_pnl);
    const unrealizedPnl = Number(row.unrealized_pnl);
    const contract = Number.isFinite(Number(row.strike_price))
      ? `${formatMoney(row.strike_price)} ${row.option_type || ""}`.trim()
      : (row.option_type || "-");
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${formatDateTime(row.created_at)}</td>
      <td>${row.symbol}</td>
      <td>${contract}</td>
      <td>${row.order_kind || "-"}</td>
      <td>${row.side || "-"}</td>
      <td>${row.quantity ?? "-"}</td>
      <td>${formatMoney(row.price)}</td>
      <td>${formatMoney(row.trigger_price)}</td>
      <td class="${Number.isFinite(realizedPnl) ? (realizedPnl >= 0 ? "pnl-profit" : "pnl-loss") : ""}">${formatSignedMoney(row.realized_pnl)}</td>
      <td class="${Number.isFinite(unrealizedPnl) ? (unrealizedPnl >= 0 ? "pnl-profit" : "pnl-loss") : ""}">${formatSignedMoney(row.unrealized_pnl)}</td>
      <td>${row.status || "-"}</td>
      <td>${row.broker_order_id || "-"}</td>
      <td>${row.exit_reason || row.consensus_reason || "-"}</td>
    `;
    ordersTable.appendChild(tr);
  });
}

function renderMiniSparkline(points) {
  miniChart.innerHTML = "";
  if (!points || !points.length) {
    miniChart.innerHTML = '<text x="20" y="60" fill="#8b949e">No premium history available</text>';
    return;
  }
  const values = points.map((item) => Number(item.premium)).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const path = values.map((value, index) => {
    const x = (index / Math.max(1, values.length - 1)) * 400;
    const y = 110 - (((value - min) / range) * 90);
    return `${index === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");
  miniChart.innerHTML = `
    <path d="${path}" fill="none" stroke="#1a6fff" stroke-width="3" />
    <line x1="0" y1="110" x2="400" y2="110" stroke="#30363d" />
  `;
}

function openTradeModal(trade) {
  modalTitle.textContent = `${trade.symbol} ${trade.strike} ${trade.option_type}`;
  const detailItems = [
    ["Expiry", trade.expiry],
    ["Entry Time", formatDateTime(trade.entry_time)],
    ["Exit Time", formatDateTime(trade.exit_time)],
    ["Entry Premium", formatMoney(trade.entry_premium)],
    ["Exit Premium", formatMoney(trade.exit_premium)],
    ["Quantity", trade.quantity],
    ["P&L", formatSignedMoney(trade.realized_pnl)],
    ["ML Confidence", formatPct(Number(trade.ml_confidence || 0) * 100)],
    ["AI Score", formatMoney(trade.ai_score)],
    ["Pine Signal", trade.pine_signal || "-"],
    ["Exit Reason", trade.exit_reason || "-"]
  ];
  modalMeta.innerHTML = detailItems.map(([label, value]) => `
    <div class="detail-item">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");
  renderMiniSparkline(trade.premium_history || []);
  tradeModal.classList.remove("hidden");
}

function closeTradeModal() {
  tradeModal.classList.add("hidden");
}

function invalidateGlobalSnapshot() {
  state.globalSnapshot = null;
  state.globalSnapshotAt = 0;
}

async function closePosition(positionId) {
  await fetchJSON(`/api/positions/${positionId}/close`, { method: "POST" });
  invalidateGlobalSnapshot();
  await loadCore({ forceGlobal: true });
}

async function loadHealth() {
  try {
    const payload = await fetchJSON(`/health?_ts=${Date.now()}`);
    apiStatus.textContent = payload.status || "ok";
    apiStatus.style.color = "#00c853";
  } catch (error) {
    apiStatus.textContent = "down";
    apiStatus.style.color = "#ff3d00";
  }
}

async function loadSymbols() {
  const fallbackSymbols = ["Nifty 50", "Bank Nifty", "SENSEX", "India VIX"];
  let symbols = fallbackSymbols;
  try {
    const payload = await fetchJSON(`/data/chart/options?_ts=${Date.now()}`);
    if (Array.isArray(payload.symbols) && payload.symbols.length) {
      symbols = payload.symbols;
    }
  } catch (error) {
    console.error(error);
    headerSubtext.textContent = "Symbol list degraded. Using configured defaults.";
  }
  const previous = state.symbol;
  symbolSelect.innerHTML = "";
  symbols.forEach((symbol) => {
    const option = document.createElement("option");
    option.value = symbol;
    option.textContent = symbol;
    symbolSelect.appendChild(option);
  });
  if (previous && symbols.some((symbol) => sameSymbol(symbol, previous))) {
    symbolSelect.value = symbols.find((symbol) => sameSymbol(symbol, previous)) || previous;
  }
  state.symbol = symbolSelect.value || symbols[0] || "Nifty 50";
}

async function loadGlobalData(force = false) {
  const now = Date.now();
  if (!force && state.globalSnapshot && (now - state.globalSnapshotAt) < 2000) {
    return state.globalSnapshot;
  }
  const fallback = state.globalSnapshot || defaultGlobalSnapshot();
  const [stats, positions, history] = await Promise.allSettled([
    fetchJSON(`/api/dashboard/stats?_ts=${now}`),
    fetchJSON(`/api/positions/open?_ts=${now}`),
    fetchJSON(`/api/positions/history?_ts=${now}`)
  ]);
  state.globalSnapshot = {
    stats: settledValue(stats, fallback.stats),
    positions: settledValue(positions, fallback.positions),
    history: settledValue(history, fallback.history),
    _errors: [
      settledError(stats, "stats"),
      settledError(positions, "positions"),
      settledError(history, "history")
    ].filter(Boolean)
  };
  state.globalSnapshotAt = now;
  return state.globalSnapshot;
}

async function loadSymbolData(symbol) {
  const stamp = Date.now();
  const chartUrl = `/data/chart?symbol=${encodeURIComponent(symbol)}&interval=1minute&prediction_target_mode=standard&candles_limit=1200&_ts=${stamp}`;
  const optionUrl = `/options/signal?symbol=${encodeURIComponent(symbol)}&interval=1minute&prediction_mode=standard&strike_mode=auto&strategy_mode=auto&allow_option_writing=false&_ts=${stamp}`;
  const cacheKey = normalizeSymbol(symbol);
  const fallback = state.symbolSnapshots[cacheKey] || defaultSymbolSnapshot(symbol);
  const [chart, consensus, logs, optionPayload, orders] = await Promise.allSettled([
    fetchJSON(chartUrl),
    fetchJSON(`/api/consensus/live?symbol=${encodeURIComponent(symbol)}&_ts=${stamp}`),
    fetchJSON(`/api/signals/log?symbol=${encodeURIComponent(symbol)}&limit=50&_ts=${stamp}`),
    fetchJSON(optionUrl),
    fetchJSON(`/api/orders/recent?symbol=${encodeURIComponent(symbol)}&limit=50&_ts=${stamp}`)
  ]);
  const snapshot = {
    chart: settledValue(chart, fallback.chart),
    consensus: settledValue(consensus, fallback.consensus),
    logs: settledValue(logs, fallback.logs),
    optionPayload: settledValue(optionPayload, fallback.optionPayload),
    orders: settledValue(orders, fallback.orders),
    _errors: [
      settledError(chart, "chart"),
      settledError(consensus, "consensus"),
      settledError(logs, "signal log"),
      settledError(optionPayload, "options"),
      settledError(orders, "orders")
    ].filter(Boolean)
  };
  state.symbolSnapshots[cacheKey] = snapshot;
  return snapshot;
}

async function loadCore({ forceGlobal = false } = {}) {
  if (!state.symbol) {
    state.symbol = symbolSelect.value || "Nifty 50";
  }
  const requestSeq = ++state.requestSeq;
  headerSubtext.textContent = `Loading ${state.symbol} execution state...`;
  try {
    const [globalData, symbolData] = await Promise.all([
      loadGlobalData(forceGlobal),
      loadSymbolData(state.symbol)
    ]);
    if (requestSeq !== state.requestSeq) {
      return;
    }

    const positions = globalData.positions || [];
    const positionsForSymbol = positions.filter((row) => sameSymbol(row.symbol, state.symbol));
    state.latestOptionSignal = symbolData.optionPayload;
    renderStats(globalData.stats || defaultStats());
    renderChart(symbolData.chart || defaultChartPayload(state.symbol));
    renderConsensus(symbolData.consensus || defaultConsensus(state.symbol), symbolData.optionPayload || defaultOptionPayload(), positionsForSymbol);
    renderOpenPositions(positions);
    renderSignalLog(symbolData.logs || []);
    renderOptionsDesk(symbolData.optionPayload || defaultOptionPayload(), positionsForSymbol, symbolData.consensus || defaultConsensus(state.symbol));
    renderList(mlReasons, symbolData.consensus?.ml_reasons || []);
    renderList(aiReasons, symbolData.consensus?.ai_reasons || []);
    renderConsensusDetails(symbolData.consensus || defaultConsensus(state.symbol));
    renderHistory(globalData.history || []);
    renderOrders(symbolData.orders || []);
    const errors = [...(globalData._errors || []), ...(symbolData._errors || [])];
    const degraded = errors.length ? ` · degraded: ${errors.join(" · ")}` : "";
    headerSubtext.textContent = `IST refresh ${formatIstTime(new Date())}${degraded}`;
  } catch (error) {
    console.error(error);
    if (requestSeq !== state.requestSeq) {
      return;
    }
    headerSubtext.textContent = `Unable to load ${state.symbol}: ${error.message || "unknown error"}`;
  }
}

async function loadCalendarDay(dateValue) {
  const payload = await fetchJSON(`/api/calendar/day/${dateValue}?_ts=${Date.now()}`);
  const summary = payload.daily_summary;
  const trades = payload.trades || [];
  calendarDayPanel.innerHTML = `
    <div class="panel-head">
      <div>
        <h2>${payload.date}</h2>
        <p>${summary.total_trades} trades · ${formatSignedMoney(summary.total_pnl)} · Win rate ${formatPct(summary.win_rate)}</p>
      </div>
    </div>
    <div class="table-shell">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Symbol</th>
            <th>Strike</th>
            <th>Type</th>
            <th>Entry</th>
            <th>Exit</th>
            <th>Qty</th>
            <th>P&amp;L</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          ${trades.length ? trades.map((trade) => `
            <tr data-calendar-trade='${JSON.stringify(trade).replace(/'/g, "&apos;")}'>
              <td>${formatDateTime(trade.entry_time)}</td>
              <td>${trade.symbol}</td>
              <td>${trade.strike}</td>
              <td>${trade.option_type}</td>
              <td>${formatMoney(trade.entry_premium)}</td>
              <td>${formatMoney(trade.exit_premium)}</td>
              <td>${trade.quantity}</td>
              <td class="${Number(trade.realized_pnl) >= 0 ? "pnl-profit" : "pnl-loss"}">${formatSignedMoney(trade.realized_pnl)}</td>
              <td>${trade.exit_reason || "-"}</td>
            </tr>
          `).join("") : '<tr><td colspan="9" class="empty-state">No trades for this day.</td></tr>'}
        </tbody>
      </table>
    </div>
  `;
}

async function loadCalendar() {
  const { year, month } = state.calendarDate;
  const payload = await fetchJSON(`/api/calendar/monthly?year=${year}&month=${month}&_ts=${Date.now()}`);
  const monthName = new Date(year, month - 1, 1).toLocaleString("en-IN", { month: "long", year: "numeric" });
  calendarMonthLabel.textContent = monthName;

  const summary = payload.reduce((acc, day) => {
    acc.totalPnl += Number(day.total_pnl || 0);
    if (Number(day.total_pnl || 0) > 0) acc.greenDays += 1;
    if (Number(day.total_pnl || 0) < 0) acc.redDays += 1;
    acc.best = Math.max(acc.best, Number(day.total_pnl || 0));
    acc.worst = Math.min(acc.worst, Number(day.total_pnl || 0));
    return acc;
  }, { totalPnl: 0, greenDays: 0, redDays: 0, best: 0, worst: 0 });
  calendarSummary.innerHTML = `
    <span>Total P&amp;L ${formatSignedMoney(summary.totalPnl)}</span>
    <span>Winning Days ${summary.greenDays}</span>
    <span>Losing Days ${summary.redDays}</span>
    <span>Best Day ${formatSignedMoney(summary.best)}</span>
    <span>Worst Day ${formatSignedMoney(summary.worst)}</span>
  `;

  const map = new Map(payload.map((row) => [row.date, row]));
  const first = new Date(year, month - 1, 1);
  const last = new Date(year, month, 0);
  const leading = (first.getDay() + 6) % 7;
  const cells = [];
  for (let index = 0; index < leading; index += 1) {
    cells.push({ muted: true });
  }
  for (let day = 1; day <= last.getDate(); day += 1) {
    const iso = `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
    const row = map.get(iso);
    cells.push({
      date: iso,
      day,
      total_pnl: row?.total_pnl,
      total_trades: row?.total_trades || 0,
      is_green: row?.is_green
    });
  }
  calendarGrid.innerHTML = cells.map((cell) => {
    if (cell.muted) {
      return '<div class="calendar-cell muted"></div>';
    }
    const cls = Number(cell.total_pnl || 0) > 0 ? "green" : (Number(cell.total_pnl || 0) < 0 ? "red" : "");
    return `
      <button class="calendar-cell ${cls}" data-calendar-date="${cell.date}">
        <strong>${cell.day}</strong>
        <span>${cell.total_trades ? `${cell.total_trades} trades` : "No trades"}</span>
        <strong>${cell.total_trades ? formatSignedMoney(cell.total_pnl) : "-"}</strong>
      </button>
    `;
  }).join("");
  calendarDayPanel.innerHTML = '<div class="empty-state">Select a trading day to see details.</div>';
}

async function emergencyStop() {
  if (!window.confirm("Close all open positions immediately?")) {
    return;
  }
  await fetchJSON("/api/emergency-stop", { method: "POST" });
  invalidateGlobalSnapshot();
  await loadCore({ forceGlobal: true });
}

navButtons.forEach((button) => {
  button.addEventListener("click", () => activateView(button.dataset.view));
});

symbolSelect.addEventListener("change", async () => {
  state.symbol = symbolSelect.value;
  await loadCore({ forceGlobal: false });
});

prevMonthBtn.addEventListener("click", async () => {
  state.calendarDate.month -= 1;
  if (state.calendarDate.month === 0) {
    state.calendarDate.month = 12;
    state.calendarDate.year -= 1;
  }
  await loadCalendar();
});

nextMonthBtn.addEventListener("click", async () => {
  state.calendarDate.month += 1;
  if (state.calendarDate.month === 13) {
    state.calendarDate.month = 1;
    state.calendarDate.year += 1;
  }
  await loadCalendar();
});

document.body.addEventListener("click", async (event) => {
  const closeTarget = event.target.closest("[data-close-position]");
  if (closeTarget) {
    const positionId = Number(closeTarget.dataset.closePosition);
    if (Number.isFinite(positionId)) {
      await closePosition(positionId);
    }
    return;
  }

  const tradeRow = event.target.closest("tr[data-trade]");
  if (tradeRow) {
    openTradeModal(JSON.parse(tradeRow.dataset.trade));
    return;
  }

  const calendarButton = event.target.closest("[data-calendar-date]");
  if (calendarButton) {
    await loadCalendarDay(calendarButton.dataset.calendarDate);
    return;
  }

  const calendarTrade = event.target.closest("tr[data-calendar-trade]");
  if (calendarTrade) {
    openTradeModal(JSON.parse(calendarTrade.dataset.calendarTrade.replace(/&apos;/g, "'")));
    return;
  }

  if (event.target.closest("[data-close-modal='true']")) {
    closeTradeModal();
  }
});

emergencyStopBtn.addEventListener("click", emergencyStop);

(async function init() {
  createChart();
  updateClock();
  setInterval(updateClock, 1000);
  try {
    await loadHealth();
    await loadSymbols();
    await loadCore({ forceGlobal: true });
  } catch (error) {
    console.error(error);
    headerSubtext.textContent = `Startup load failed: ${error.message || "unknown error"}`;
  }
  setInterval(async () => {
    if (!document.hidden) {
      try {
        await loadHealth();
        await loadCore({ forceGlobal: true });
      } catch (error) {
        console.error(error);
        headerSubtext.textContent = `Auto refresh degraded: ${error.message || "unknown error"}`;
      }
    }
  }, 5000);
})();
