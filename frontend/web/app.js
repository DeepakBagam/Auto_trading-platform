const { startTransition, useDeferredValue, useEffect, useRef, useState } = React;
const { createRoot } = ReactDOM;
const { LightweightCharts } = window;

const NAV_ITEMS = [
  { id: "overview", label: "Overview", eyebrow: "Desk", icon: "dashboard" },
  { id: "operations", label: "Operations", eyebrow: "Controls", icon: "monitoring" },
  { id: "positions", label: "Positions", eyebrow: "Trading", icon: "account_balance_wallet" },
  { id: "optionchain", label: "Option Chain", eyebrow: "Options", icon: "rebase_edit" },
  { id: "history", label: "Trade History", eyebrow: "Analytics", icon: "history" },
  { id: "calendar", label: "Calendar", eyebrow: "Sessions", icon: "calendar_today" },
  { id: "database", label: "Database", eyebrow: "Coverage", icon: "database" },
  { id: "settings", label: "Settings", eyebrow: "Runtime", icon: "settings" },
];
const IST_TIME_ZONE = "Asia/Kolkata";
const CHART_RANGE_FALLBACK = [
  { key: "all", label: "ALL", interval: "1minute", supports_live: true },
];
const CHART_INTERVAL_FALLBACK = [
  { key: "1m", label: "1m", interval: "1minute" },
];
const LIVE_CHART_WINDOW_LIMITS = {
  "all": 500000,
};
const PREFETCH_CHART_RANGES = [];
const PREFETCH_SYMBOLS = [];
const HISTORY_BATCH_SIZE = 5000;
const FULL_HISTORY_CANDLE_LIMIT = 1000;
const TERMINAL_LAYOUT_STORAGE_KEY = "alpha-terminal-layout-v2";
const DEFAULT_INDICATORS = [
  { id: "ema-9", type: "EMA", period: 9, color: "#f7c948", enabled: true },
  { id: "ema-21", type: "EMA", period: 21, color: "#7dd3fc", enabled: true },
];
const DRAWING_TOOLS = [
  { key: "trendline", icon: "show_chart", label: "Trend line" },
  { key: "ray", icon: "trending_up", label: "Ray" },
  { key: "horizontal", icon: "horizontal_rule", label: "Horizontal line" },
  { key: "vertical", icon: "vertical_split", label: "Vertical line" },
  { key: "rectangle", icon: "crop_square", label: "Rectangle" },
  { key: "circle", icon: "radio_button_unchecked", label: "Circle" },
  { key: "arrow", icon: "arrow_forward", label: "Arrow" },
  { key: "text", icon: "title", label: "Text" },
  { key: "fib", icon: "stacked_line_chart", label: "Fibonacci" },
];
const REPLAY_SPEEDS = [1, 2, 5, 10, 50, 100];
const THEME_STORAGE_KEY = "alpha-terminal-theme";
const CHART_THEMES = {
  dark: {
    background: "#0c1322",
    text: "#dce2f7",
    border: "#424754",
    grid: "rgba(66, 71, 84, 0.45)",
    up: "#40e56c",
    down: "#ff5353",
  },
  light: {
    background: "#f6f8fc",
    text: "#172033",
    border: "#c7d1df",
    grid: "rgba(101, 116, 139, 0.24)",
    up: "#087a46",
    down: "#c43131",
  },
};

function getInitialTheme() {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") {
      return stored;
    }
    return window.matchMedia?.("(prefers-color-scheme: light)")?.matches ? "light" : "dark";
  } catch (_error) {
    return "dark";
  }
}

function formatMoney(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? amount.toFixed(2) : "-";
}

function formatSignedMoney(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) {
    return "-";
  }
  const prefix = amount > 0 ? "+" : "";
  return `${prefix}${amount.toFixed(2)}`;
}

function formatPct(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? `${amount.toFixed(2)}%` : "-";
}

function formatCount(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? amount.toLocaleString("en-IN") : "-";
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? "-"
    : parsed.toLocaleString("en-IN", { hour12: false, timeZone: IST_TIME_ZONE });
}

function formatDate(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? "-"
    : parsed.toLocaleDateString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        timeZone: IST_TIME_ZONE,
      });
}

function formatTime(value) {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? "-"
    : parsed.toLocaleTimeString("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        timeZone: IST_TIME_ZONE,
      });
}

function formatNs(value) {
  const amount = Number(value);
  if (!Number.isFinite(amount)) {
    return "-";
  }
  if (amount < 1_000) {
    return `${Math.round(amount)} ns`;
  }
  if (amount < 1_000_000) {
    return `${(amount / 1_000).toFixed(1)} us`;
  }
  if (amount < 1_000_000_000) {
    return `${(amount / 1_000_000).toFixed(1)} ms`;
  }
  return `${(amount / 1_000_000_000).toFixed(2)} s`;
}

function formatAge(seconds) {
  const amount = Number(seconds);
  if (!Number.isFinite(amount)) {
    return "-";
  }
  if (amount < 60) {
    return `${Math.round(amount)}s`;
  }
  const mins = Math.floor(amount / 60);
  const secs = Math.round(amount % 60);
  return `${mins}m ${secs}s`;
}

function formatFlag(value) {
  return value ? "Yes" : "No";
}

function actionTone(action) {
  const normalized = String(action || "").toUpperCase();
  if (normalized === "BUY") {
    return "buy";
  }
  if (normalized === "SELL") {
    return "sell";
  }
  return "hold";
}

function signalTone(signal) {
  const action = String(signal?.action || "").toUpperCase();
  return actionTone(action === "HOLD" ? signal?.bias : action);
}

function signalLabel(signal) {
  const action = String(signal?.action || "HOLD").toUpperCase();
  const bias = String(signal?.bias || "").toUpperCase();
  if (action === "HOLD" && (bias === "BUY" || bias === "SELL")) {
    return `${bias} bias`;
  }
  return action;
}

function parseBrokerError(value) {
  if (!value) {
    return "";
  }
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      const first = Array.isArray(parsed.errors) ? parsed.errors[0] : null;
      return first?.message || parsed.message || value;
    } catch (_error) {
      return value;
    }
  }
  const first = Array.isArray(value.errors) ? value.errors[0] : null;
  return first?.message || value.message || value.error || "";
}

function entryReadiness(signal, optionSignal) {
  const optionAction = String(optionSignal?.action || "").toUpperCase();
  const signalAction = String(signal?.action || "").toUpperCase();
  if (optionAction === "BUY" || optionAction === "SELL") {
    return { label: "Ready", tone: "buy" };
  }
  if (signalAction === "BUY" || signalAction === "SELL") {
    return { label: "Waiting Contract", tone: "hold" };
  }
  return { label: "Waiting", tone: "hold" };
}

function riskReward(entry, stopLoss, target) {
  const entryValue = Number(entry);
  const stopValue = Number(stopLoss);
  const targetValue = Number(target);
  if (![entryValue, stopValue, targetValue].every(Number.isFinite) || entryValue === stopValue) {
    return "-";
  }
  const risk = Math.abs(entryValue - stopValue);
  const reward = Math.abs(targetValue - entryValue);
  if (risk <= 0) {
    return "-";
  }
  return `${(reward / risk).toFixed(2)}R`;
}

function parseChartTime(value) {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : Math.floor(parsed.getTime() / 1000);
}

function chartTimeToDate(timeValue) {
  if (typeof timeValue === "number") {
    return new Date(timeValue * 1000);
  }
  if (timeValue && typeof timeValue === "object" && Number.isFinite(timeValue.year)) {
    return new Date(Date.UTC(timeValue.year, (timeValue.month || 1) - 1, timeValue.day || 1));
  }
  return null;
}

function formatChartTickMark(timeValue) {
  const parsed = chartTimeToDate(timeValue);
  if (!parsed) {
    return "";
  }
  const istParts = new Intl.DateTimeFormat("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: IST_TIME_ZONE,
  }).formatToParts(parsed);
  const hour = Number(istParts.find((part) => part.type === "hour")?.value);
  const minute = Number(istParts.find((part) => part.type === "minute")?.value);
  if (hour === 0 && minute === 0) {
    return parsed.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      timeZone: IST_TIME_ZONE,
    });
  }
  return parsed.toLocaleTimeString("en-IN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    timeZone: IST_TIME_ZONE,
  });
}

function formatChartCrosshairTime(timeValue) {
  const parsed = chartTimeToDate(timeValue);
  return parsed
    ? parsed.toLocaleString("en-IN", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
        timeZone: IST_TIME_ZONE,
      })
    : "";
}

function chartCacheKey(symbol, rangeKey, intervalKey) {
  return `${symbol || ""}::all::1minute`;
}

function normalizeChartCandles(rows = []) {
  return rows
    .map((row) => ({
      time: parseChartTime(row.x),
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
      volume: Number(row.volume || 0),
    }))
    .filter((row) => (
      Number.isFinite(row.time)
      && Number.isFinite(row.open)
      && Number.isFinite(row.high)
      && Number.isFinite(row.low)
      && Number.isFinite(row.close)
    ));
}

function chartCandleKey(row) {
  return row?.x || row?.time || "";
}

function mergeChartCandleRows(existing = [], incoming = [], mode = "append") {
  const merged = mode === "prepend" ? [...incoming, ...existing] : [...existing, ...incoming];
  const byTime = new Map();
  merged.forEach((row) => {
    const key = chartCandleKey(row);
    if (key) {
      byTime.set(key, row);
    }
  });
  return Array.from(byTime.values()).sort((a, b) => parseChartTime(chartCandleKey(a)) - parseChartTime(chartCandleKey(b)));
}

function chartEndpointParams(symbol, rangeKey, intervalKey) {
  return `symbol=${encodeURIComponent(symbol)}&range=all&interval=1minute`;
}

function chartFromCandlePayload(payload, rangeKey = "all") {
  const oldest = payload?.oldest || payload?.candles?.[0]?.x || null;
  const latest = payload?.latest || payload?.candles?.[payload?.candles?.length - 1]?.x || null;
  return {
    symbol: payload?.symbol,
    instrument_key: payload?.instrument_key,
    range: rangeKey,
    label: rangeKey === "all" ? "ALL" : rangeKey.toUpperCase(),
    interval: payload?.interval || "1minute",
    supports_live: payload?.interval === "1minute",
    is_resampled: payload?.interval !== "1minute",
    start_date: oldest ? oldest.slice(0, 10) : null,
    end_date: latest ? latest.slice(0, 10) : null,
    generated_at: new Date().toISOString(),
    candles: payload?.candles || [],
    available_count: Number(payload?.available_count || payload?.total || 0) || null,
    oldest,
    latest,
    markers: [],
    available_ranges: CHART_RANGE_FALLBACK,
    available_intervals: payload?.available_intervals || CHART_INTERVAL_FALLBACK,
  };
}

async function fetchChartPayload(symbol, rangeKey, intervalKey) {
  const payload = await apiChartFetch(
    `/api/candles?symbol=${encodeURIComponent(symbol)}&interval=1minute&limit=${FULL_HISTORY_CANDLE_LIMIT}`,
  );
  return chartFromCandlePayload(payload, "all");
}

function stableId(prefix) {
  return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function readStoredLayout() {
  try {
    const raw = window.localStorage.getItem(TERMINAL_LAYOUT_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    if (parsed && typeof parsed === "object") {
      return parsed;
    }
  } catch (_error) {
    // Ignore corrupt layout; defaults keep the terminal usable.
  }
  return {};
}

function writeStoredLayout(layout) {
  try {
    window.localStorage.setItem(TERMINAL_LAYOUT_STORAGE_KEY, JSON.stringify(layout));
  } catch (_error) {
    // Storage quota failures should not block trading workflow.
  }
}

function defaultLayout() {
  return {
    selectedSymbol: "",
    timeframe: "1minute",
    range: "all",
    theme: getInitialTheme(),
    recentSearches: [],
    favorites: [],
    activeWatchlist: "Indices",
    watchlists: {
      Indices: ["Nifty 50", "Bank Nifty", "SENSEX", "India VIX"],
      Stocks: [],
      "F&O": [],
      Crypto: [],
    },
    templates: {},
    alerts: [],
    symbolState: {},
  };
}

function mergeLayout(raw) {
  const defaults = defaultLayout();
  return {
    ...defaults,
    ...raw,
    watchlists: { ...defaults.watchlists, ...(raw.watchlists || {}) },
    templates: { ...(raw.templates || {}) },
    alerts: Array.isArray(raw.alerts) ? raw.alerts : [],
    recentSearches: Array.isArray(raw.recentSearches) ? raw.recentSearches : [],
    favorites: Array.isArray(raw.favorites) ? raw.favorites : [],
    symbolState: raw.symbolState && typeof raw.symbolState === "object" ? raw.symbolState : {},
  };
}

function defaultSymbolState() {
  return {
    range: "all",
    interval: "1minute",
    indicators: DEFAULT_INDICATORS,
    drawings: [],
    visibleRange: null,
    chartSettings: { volume: true, vwap: false, chartType: "candles" },
  };
}

function symbolStateKey(symbol) {
  return String(symbol || "default").replace(/\s+/g, "").toUpperCase();
}

function getSymbolState(layout, symbol) {
  const key = symbolStateKey(symbol);
  return {
    ...defaultSymbolState(),
    ...(layout.symbolState?.[key] || {}),
  };
}

function upsertSymbolState(layout, symbol, patch) {
  const key = symbolStateKey(symbol);
  const previous = getSymbolState(layout, symbol);
  return {
    ...layout,
    symbolState: {
      ...(layout.symbolState || {}),
      [key]: {
        ...previous,
        ...patch,
      },
    },
  };
}

function addUnique(list, value, limit = 12) {
  const raw = String(value || "").trim();
  if (!raw) {
    return list || [];
  }
  return [raw, ...(list || []).filter((item) => item !== raw)].slice(0, limit);
}

function buildMovingAverage(candles, period, mode = "EMA") {
  const windowSize = Math.max(1, Number(period) || 1);
  if (mode === "SMA") {
    let running = 0;
    return candles.map((row, index) => {
      running += row.close;
      if (index >= windowSize) {
        running -= candles[index - windowSize].close;
      }
      return index + 1 >= windowSize ? { time: row.time, value: Number((running / windowSize).toFixed(2)) } : null;
    }).filter(Boolean);
  }
  const multiplier = 2 / (windowSize + 1);
  let ema = null;
  return candles.map((row, index) => {
    ema = index === 0 ? row.close : ((row.close - ema) * multiplier) + ema;
    return index + 1 >= windowSize ? { time: row.time, value: Number(ema.toFixed(2)) } : null;
  }).filter(Boolean);
}

function buildBollingerSeries(candles, period = 20, multiplier = 2) {
  const windowSize = Math.max(2, Number(period) || 20);
  const mult = Math.max(0.1, Number(multiplier) || 2);
  return candles.map((row, index) => {
    if (index + 1 < windowSize) {
      return null;
    }
    const slice = candles.slice(index + 1 - windowSize, index + 1);
    const mean = slice.reduce((sum, item) => sum + item.close, 0) / windowSize;
    const variance = slice.reduce((sum, item) => sum + ((item.close - mean) ** 2), 0) / windowSize;
    const deviation = Math.sqrt(variance);
    return {
      time: row.time,
      upper: Number((mean + (deviation * mult)).toFixed(2)),
      middle: Number(mean.toFixed(2)),
      lower: Number((mean - (deviation * mult)).toFixed(2)),
    };
  }).filter(Boolean);
}

function buildAtrSeries(candles, period = 14) {
  const windowSize = Math.max(1, Number(period) || 14);
  const trueRanges = candles.map((row, index) => {
    const previousClose = candles[index - 1]?.close ?? row.close;
    return Math.max(row.high - row.low, Math.abs(row.high - previousClose), Math.abs(row.low - previousClose));
  });
  return trueRanges.map((value, index) => {
    if (index + 1 < windowSize) {
      return null;
    }
    const slice = trueRanges.slice(index + 1 - windowSize, index + 1);
    return { time: candles[index].time, value: Number((slice.reduce((sum, item) => sum + item, 0) / windowSize).toFixed(2)) };
  }).filter(Boolean);
}

function buildSupertrendSeries(candles, period = 10, multiplier = 3) {
  const atr = buildAtrSeries(candles, period);
  const atrByTime = new Map(atr.map((row) => [row.time, row.value]));
  return candles.map((row) => {
    const value = atrByTime.get(row.time);
    if (!value) {
      return null;
    }
    const mid = (row.high + row.low) / 2;
    const directionUp = row.close >= row.open;
    const line = directionUp ? mid - value * multiplier : mid + value * multiplier;
    return { time: row.time, value: Number(line.toFixed(2)) };
  }).filter(Boolean);
}

function buildIndicatorSeries(indicator, candles) {
  if (!indicator?.enabled) {
    return [];
  }
  const type = String(indicator.type || "").toUpperCase();
  if (type === "EMA" || type === "SMA") {
    return buildMovingAverage(candles, indicator.period || 20, type);
  }
  if (type === "VWAP") {
    return buildVwapSeries(candles);
  }
  if (type === "BB") {
    return buildBollingerSeries(candles, indicator.period || 20, indicator.multiplier || 2);
  }
  if (type === "ATR") {
    return buildAtrSeries(candles, indicator.period || 14);
  }
  if (type === "SUPERTREND") {
    return buildSupertrendSeries(candles, indicator.period || 10, indicator.multiplier || 3);
  }
  return [];
}

function indicatorLabel(indicator) {
  const type = String(indicator.type || "").toUpperCase();
  if (type === "VWAP") {
    return "VWAP";
  }
  if (type === "BB") {
    return `BB ${indicator.period || 20}`;
  }
  return `${type} ${indicator.period || ""}`.trim();
}

function buildEmaSeries(candles, period) {
  const multiplier = 2 / (period + 1);
  let ema = null;
  return candles
    .map((row, index) => {
      ema = index === 0 ? row.close : ((row.close - ema) * multiplier) + ema;
      return index + 1 >= period ? { time: row.time, value: Number(ema.toFixed(2)) } : null;
    })
    .filter(Boolean);
}

function buildVwapSeries(candles) {
  let cumulativeTypicalVolume = 0;
  let cumulativeVolume = 0;
  return candles
    .map((row) => {
      const volume = Math.max(0, Number(row.volume || 0));
      if (!volume) {
        return null;
      }
      const typical = (row.high + row.low + row.close) / 3;
      cumulativeTypicalVolume += typical * volume;
      cumulativeVolume += volume;
      return { time: row.time, value: Number((cumulativeTypicalVolume / cumulativeVolume).toFixed(2)) };
    })
    .filter(Boolean);
}

function candleChangePct(row) {
  if (!row || !Number.isFinite(row.open) || row.open === 0) {
    return "-";
  }
  return formatPct(((row.close - row.open) / row.open) * 100);
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Request failed");
  }
  return response.json();
}

function apiChartFetch(url) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url, true);
    request.timeout = 120000;
    request.onload = () => {
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(request.responseText || "Chart request failed"));
        return;
      }
      try {
        resolve(JSON.parse(request.responseText));
      } catch (error) {
        reject(error);
      }
    };
    request.onerror = () => reject(new Error("Chart request failed"));
    request.ontimeout = () => reject(new Error("Chart request timed out"));
    request.send();
  });
}

function mergeQuickUpdate(currentSnapshot, update) {
  const current = currentSnapshot || {};
  return {
    ...current,
    generated_at: update?.generated_at || current.generated_at,
    symbol: update?.symbol || current.symbol,
    instrument_key: update?.instrument_key || current.instrument_key,
    price: update?.price || current.price || {},
    freshness: update?.freshness || current.freshness || {},
    stream: update?.stream || current.stream || {},
  };
}

function mergeLiveChart(currentChart, update) {
  const current = currentChart || {};
  if (!current.supports_live) {
    return current;
  }
  const nextCandles = [...(current.candles || [])];
  const nextCandle = update?.candle;

  if (nextCandle?.x) {
    const lastIndex = nextCandles.length - 1;
    if (lastIndex >= 0 && nextCandles[lastIndex]?.x === nextCandle.x) {
      nextCandles[lastIndex] = nextCandle;
    } else {
      nextCandles.push(nextCandle);
    }
  }

  const configuredWindow = LIVE_CHART_WINDOW_LIMITS[current.range] || 0;
  const currentWindow = Math.max((current.candles || []).length, 1);
  const windowSize = Math.max(configuredWindow, currentWindow);
  return {
    ...current,
    generated_at: update?.generated_at || current.generated_at,
    candles: nextCandles.slice(-windowSize),
  };
}

function MetricCard({ label, value, meta, tone }) {
  return (
    <article className="metric-card">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
      <small>{meta}</small>
    </article>
  );
}

function InlineLoader({ label = "Loading..." }) {
  return <div className="inline-loader">{label}</div>;
}

function Sidebar({ activeView, onChange, snapshot, streamState }) {
  const calendar = snapshot?.calendar || {};
  const history = snapshot?.history || {};
  const notifications = snapshot?.notifications || {};
  const optionSignal = snapshot?.option?.signal || {};
  const engineAction = optionSignal.action || snapshot?.signal?.action || "HOLD";
  const engineTone = actionTone(engineAction);
  const selectedStrike = optionSignal.strike || optionSignal.strike_price || "-";
  const selectedType = optionSignal.option_type || "-";
  const streamTone = streamState === "live" ? "live" : (streamState === "reconnecting" ? "warn" : "down");
  const sourceRunning = Boolean(snapshot?.stream?.runtime?.running);
  const marketStatus = snapshot?.freshness?.market_status || "unknown";
  const sourceState = marketStatus === "live" ? "live" : (sourceRunning ? marketStatus : "stopped");
  const sourceTone = sourceState === "live" ? "live" : (sourceState === "stopped" ? "down" : "warn");
  const mailState = !notifications.smtp_enabled
    ? "off"
    : (notifications.smtp_ready ? "ready" : "incomplete");

  return (
    <aside className="sidebar">
      <div className="brand-card">
        <div className="brand-mark">AT</div>
        <div>
          <h1>Alpha Terminal</h1>
          <p>Institutional Grade</p>
        </div>
      </div>

      <nav className="nav-card">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-button ${activeView === item.id ? "active" : ""}`}
            onClick={() => onChange(item.id)}
          >
            <span className="material-symbols-outlined" aria-hidden="true">{item.icon}</span>
            <span>
              <strong>{item.label}</strong>
              <small>{item.eyebrow}</small>
            </span>
          </button>
        ))}
      </nav>

      <div className="sidebar-card engine-sidebar-card">
        <h3>Engine Strike</h3>
        <div className={`engine-mini ${engineTone}`}>
          <div className="engine-mini-top">
            <strong>{engineAction || "HOLD"}</strong>
            <span>{selectedType}</span>
          </div>
          <div className="engine-mini-strike">{selectedStrike}</div>
          <div className="mini-row">
            <span>Entry</span>
            <strong>{formatMoney(optionSignal.entry_price)}</strong>
          </div>
          <div className="mini-row">
            <span>SL / TP</span>
            <strong>{formatMoney(optionSignal.stop_loss)} / {formatMoney(optionSignal.take_profit)}</strong>
          </div>
        </div>
      </div>

      <div className="sidebar-card system-sidebar-card">
        <h3>System</h3>
        <div className="mini-list">
          <div className="mini-row">
            <span>Socket</span>
            <strong><span className={`pill-dot ${streamTone}`} /> {streamState}</strong>
          </div>
          <div className="mini-row">
            <span>Source</span>
            <strong><span className={`pill-dot ${sourceTone}`} /> {sourceState}</strong>
          </div>
          <div className="mini-row">
            <span>Today IST</span>
            <strong>{formatDate(calendar.today_ist)}</strong>
          </div>
          <div className="mini-row">
            <span>Retention</span>
            <strong>{history.retention_years || 2} years</strong>
          </div>
          <div className="mini-row">
            <span>Mail</span>
            <strong>{mailState}</strong>
          </div>
          <div className="mini-row">
            <span>Recipients</span>
            <strong>{formatCount(notifications.recipient_count || 0)}</strong>
          </div>
        </div>
      </div>
    </aside>
  );
}

function ChartPanel({
  symbol,
  chart,
  rangeKey,
  intervalKey,
  indicators,
  drawings,
  activeDrawingTool,
  selectedDrawingId,
  replay,
  alerts,
  onRangeChange,
  onIntervalChange,
  onRangeWarm,
  onLoadMoreHistory,
  onIndicatorsChange,
  onDrawingsChange,
  onActiveDrawingToolChange,
  onSelectedDrawingIdChange,
  onReplayChange,
  onCreateAlert,
  onSaveTemplate,
  onApplyTemplate,
  theme,
}) {
  const hostRef = useRef(null);
  const shellRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const emaFastRef = useRef(null);
  const emaSlowRef = useRef(null);
  const vwapRef = useRef(null);
  const indicatorSeriesRef = useRef(new Map());
  const drawingDraftRef = useRef(null);
  const dragDrawingRef = useRef(null);
  const viewportStateRef = useRef({ key: "", firstTime: null, lastTime: null, candleCount: 0 });
  const historyRequestRef = useRef("");
  const [indicatorType, setIndicatorType] = useState("EMA");
  const [indicatorPeriod, setIndicatorPeriod] = useState(20);
  const [replayStart, setReplayStart] = useState("");
  const [alertPrice, setAlertPrice] = useState("");
  const [alertOperator, setAlertOperator] = useState(">");
  const [templateName, setTemplateName] = useState("");
  const [chartType, setChartType] = useState("candles");
  const [showVolume, setShowVolume] = useState(true);
  const [showEma, setShowEma] = useState(false);
  const [showVwap, setShowVwap] = useState(false);
  const [crosshairMode, setCrosshairMode] = useState("crosshair");
  const [openPanel, setOpenPanel] = useState("");
  const [crosshair, setCrosshair] = useState(null);
  const deferredChart = useDeferredValue(chart);
  const ranges = deferredChart?.available_ranges || CHART_RANGE_FALLBACK;
  const intervals = deferredChart?.available_intervals || CHART_INTERVAL_FALLBACK;
  const rangeKeys = new Set(["all", "1d", "5d", "1mo", "6mo", "1y", "2y", "5y"]);
  const primaryRanges = ranges.filter((item) => rangeKeys.has(item.key));
  const candles = React.useMemo(() => normalizeChartCandles(deferredChart?.candles || []), [deferredChart]);
  const replayIndex = replay?.active ? Math.max(0, Math.min(Number(replay.index || 0), candles.length - 1)) : null;
  const displayCandles = replay?.active ? candles.slice(0, replayIndex + 1) : candles;
  const activeCandle = crosshair || displayCandles[displayCandles.length - 1] || null;
  const activeTone = activeCandle && activeCandle.close >= activeCandle.open ? "positive" : "negative";
  const activeRangeLabel = ranges.find((item) => item.key === rangeKey)?.label || deferredChart?.label || "Chart";
  const activeIntervalLabel = intervals.find((item) => item.interval === intervalKey || item.key === intervalKey)?.label || intervalKey || "-";

  useEffect(() => {
    if (!hostRef.current || !LightweightCharts) {
      return undefined;
    }
    const chartTheme = CHART_THEMES[theme] || CHART_THEMES.dark;
    const instance = LightweightCharts.createChart(hostRef.current, {
      width: hostRef.current.clientWidth,
      height: hostRef.current.clientHeight,
      layout: { background: { color: chartTheme.background }, textColor: chartTheme.text },
      rightPriceScale: { borderColor: chartTheme.border, scaleMargins: { top: 0.08, bottom: 0.24 } },
      leftPriceScale: { visible: false },
      crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: { color: chartTheme.border, labelBackgroundColor: chartTheme.background },
        horzLine: { color: chartTheme.border, labelBackgroundColor: chartTheme.background },
      },
      timeScale: {
        borderColor: chartTheme.border,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 8,
        barSpacing: 8,
      },
      grid: {
        vertLines: { color: chartTheme.grid },
        horzLines: { color: chartTheme.grid },
      },
      localization: {
        locale: "en-IN",
        timeFormatter: formatChartCrosshairTime,
      },
    });
    instance.applyOptions({
      timeScale: {
        tickMarkFormatter: formatChartTickMark,
      },
    });
    const series = instance.addCandlestickSeries({
      upColor: chartTheme.up,
      downColor: chartTheme.down,
      wickUpColor: chartTheme.up,
      wickDownColor: chartTheme.down,
      borderVisible: false,
      lastValueVisible: true,
      priceLineVisible: true,
    });
    const volumeSeries = instance.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "",
      color: "rgba(173, 198, 255, 0.32)",
    });
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } });
    const emaFast = instance.addLineSeries({
      color: "#f7c948",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    const emaSlow = instance.addLineSeries({
      color: "#7dd3fc",
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    const vwap = instance.addLineSeries({
      color: "#f472b6",
      lineWidth: 2,
      lineStyle: LightweightCharts.LineStyle.Dotted,
      priceLineVisible: false,
      lastValueVisible: false,
    });
    chartRef.current = instance;
    seriesRef.current = series;
    volumeSeriesRef.current = volumeSeries;
    emaFastRef.current = emaFast;
    emaSlowRef.current = emaSlow;
    vwapRef.current = vwap;

    const onCrosshairMove = (param) => {
      if (!param?.time || !seriesRef.current) {
        setCrosshair(null);
        return;
      }
      const data = param.seriesData?.get(seriesRef.current);
      setCrosshair(data ? { ...data, time: param.time } : null);
    };
    instance.subscribeCrosshairMove(onCrosshairMove);

    const onResize = () => {
      if (hostRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: hostRef.current.clientWidth,
          height: hostRef.current.clientHeight,
        });
      }
    };
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("resize", onResize);
      instance.unsubscribeCrosshairMove(onCrosshairMove);
      instance.remove();
      chartRef.current = null;
      seriesRef.current = null;
      volumeSeriesRef.current = null;
      emaFastRef.current = null;
      emaSlowRef.current = null;
      vwapRef.current = null;
      indicatorSeriesRef.current.clear();
    };
  }, [theme]);

  useEffect(() => {
    if (!chartRef.current || !LightweightCharts) {
      return;
    }
    chartRef.current.applyOptions({
      crosshair: {
        mode: crosshairMode === "crosshair"
          ? LightweightCharts.CrosshairMode.Normal
          : LightweightCharts.CrosshairMode.Hidden,
      },
    });
  }, [crosshairMode]);

  useEffect(() => {
    const onFit = () => fitChart();
    const onReset = () => resetChart();
    window.addEventListener("alpha-fit-chart", onFit);
    window.addEventListener("alpha-reset-chart", onReset);
    return () => {
      window.removeEventListener("alpha-fit-chart", onFit);
      window.removeEventListener("alpha-reset-chart", onReset);
    };
  }, []);

  useEffect(() => {
    const timeScale = chartRef.current?.timeScale?.();
    if (!timeScale || typeof timeScale.subscribeVisibleLogicalRangeChange !== "function") {
      return undefined;
    }
    const onVisibleRange = (logicalRange) => {
      if (!logicalRange || !displayCandles.length || typeof onLoadMoreHistory !== "function") {
        return;
      }
      if (Number(logicalRange.from) > 40) {
        return;
      }
      const oldest = deferredChart?.oldest || deferredChart?.candles?.[0]?.x;
      const requestKey = `${symbol}:${intervalKey}:${oldest || ""}`;
      if (!oldest || historyRequestRef.current === requestKey) {
        return;
      }
      historyRequestRef.current = requestKey;
      Promise.resolve(onLoadMoreHistory(oldest)).finally(() => {
        window.setTimeout(() => {
          if (historyRequestRef.current === requestKey) {
            historyRequestRef.current = "";
          }
        }, 500);
      });
    };
    timeScale.subscribeVisibleLogicalRangeChange(onVisibleRange);
    return () => {
      timeScale.unsubscribeVisibleLogicalRangeChange?.(onVisibleRange);
    };
  }, [symbol, intervalKey, displayCandles.length, deferredChart?.oldest, deferredChart?.candles, onLoadMoreHistory]);

  useEffect(() => {
    if (!seriesRef.current || !deferredChart) {
      return;
    }
    const chartTheme = CHART_THEMES[theme] || CHART_THEMES.dark;
    seriesRef.current.applyOptions({
      upColor: chartTheme.up,
      downColor: chartTheme.down,
      wickUpColor: chartTheme.up,
      wickDownColor: chartTheme.down,
      borderVisible: chartType === "bars",
    });
    const currentKey = `${symbol || ""}:${deferredChart?.range || rangeKey}:${deferredChart?.interval || intervalKey}`;
    const previousViewport = viewportStateRef.current;
    const firstTime = displayCandles[0]?.time || null;
    const lastTime = displayCandles[displayCandles.length - 1]?.time || null;
    const sameDataset = previousViewport.key === currentKey;
    const onlyLatestChanged = (
      sameDataset
      && previousViewport.firstTime === firstTime
      && displayCandles.length >= previousViewport.candleCount
      && displayCandles.length <= previousViewport.candleCount + 1
    );
    const prependedHistory = (
      sameDataset
      && previousViewport.lastTime === lastTime
      && firstTime !== previousViewport.firstTime
      && displayCandles.length > previousViewport.candleCount
    );
    const visibleLogicalRange = chartRef.current?.timeScale?.().getVisibleLogicalRange?.();
    if (onlyLatestChanged && displayCandles.length) {
      seriesRef.current.update(displayCandles[displayCandles.length - 1]);
    } else {
      seriesRef.current.setData(displayCandles);
    }
    const volumeRows = showVolume
      ? displayCandles.map((row) => ({
          time: row.time,
          value: row.volume,
          color: row.close >= row.open ? "rgba(64, 229, 108, 0.34)" : "rgba(255, 83, 83, 0.34)",
        }))
      : [];
    if (onlyLatestChanged && volumeRows.length) {
      volumeSeriesRef.current?.update(volumeRows[volumeRows.length - 1]);
    } else {
      volumeSeriesRef.current?.setData(volumeRows);
    }
    emaFastRef.current?.setData(showEma ? buildEmaSeries(displayCandles, 9) : []);
    emaSlowRef.current?.setData(showEma ? buildEmaSeries(displayCandles, 21) : []);
    vwapRef.current?.setData(showVwap ? buildVwapSeries(displayCandles) : []);
    const activeIndicatorIds = new Set();
    (indicators || []).filter((item) => item.enabled).forEach((indicator, index) => {
      const baseId = indicator.id || `${indicator.type}-${indicator.period}-${index}`;
      const seriesRows = buildIndicatorSeries(indicator, displayCandles);
      const type = String(indicator.type || "").toUpperCase();
      if (type === "BB") {
        ["upper", "middle", "lower"].forEach((band, bandIndex) => {
          const id = `${baseId}-${band}`;
          activeIndicatorIds.add(id);
          if (!indicatorSeriesRef.current.has(id)) {
            indicatorSeriesRef.current.set(id, chartRef.current.addLineSeries({
              color: band === "middle" ? (indicator.color || "#adc6ff") : "rgba(173, 198, 255, 0.62)",
              lineWidth: bandIndex === 1 ? 1 : 2,
              priceLineVisible: false,
              lastValueVisible: false,
            }));
          }
          indicatorSeriesRef.current.get(id).setData(seriesRows.map((row) => ({ time: row.time, value: row[band] })));
        });
      } else {
        activeIndicatorIds.add(baseId);
        if (!indicatorSeriesRef.current.has(baseId)) {
          indicatorSeriesRef.current.set(baseId, chartRef.current.addLineSeries({
            color: indicator.color || "#adc6ff",
            lineWidth: 2,
            priceLineVisible: false,
            lastValueVisible: false,
          }));
        }
        indicatorSeriesRef.current.get(baseId).setData(seriesRows);
      }
    });
    Array.from(indicatorSeriesRef.current.keys()).forEach((id) => {
      if (!activeIndicatorIds.has(id)) {
        const staleSeries = indicatorSeriesRef.current.get(id);
        if (staleSeries && chartRef.current?.removeSeries) {
          chartRef.current.removeSeries(staleSeries);
        }
        indicatorSeriesRef.current.delete(id);
      }
    });
    if (chartRef.current?.timeScale) {
      if (!sameDataset) {
        chartRef.current.timeScale().fitContent();
      } else if (prependedHistory && visibleLogicalRange) {
        const added = displayCandles.length - previousViewport.candleCount;
        chartRef.current.timeScale().setVisibleLogicalRange({
          from: visibleLogicalRange.from + added,
          to: visibleLogicalRange.to + added,
        });
      }
    }
    viewportStateRef.current = { key: currentKey, firstTime, lastTime, candleCount: displayCandles.length };
    const markers = (deferredChart.markers || [])
      .map((row) => ({
        time: parseChartTime(row.time),
        position: row.position || "inBar",
        color: row.color,
        shape: row.shape || "circle",
        text: row.text || "",
      }))
      .filter((row) => Number.isFinite(row.time));
    if (typeof seriesRef.current.setMarkers === "function") {
      seriesRef.current.setMarkers(markers);
    }
  }, [deferredChart, displayCandles, indicators, theme, chartType, showVolume, showEma, showVwap]);

  function fitChart() {
    chartRef.current?.timeScale?.().fitContent();
  }

  function goLive() {
    chartRef.current?.timeScale?.().scrollToRealTime();
  }

  function zoomChart(multiplier) {
    const timeScale = chartRef.current?.timeScale?.();
    const range = timeScale?.getVisibleLogicalRange?.();
    if (!timeScale || !range) {
      return;
    }
    const center = (range.from + range.to) / 2;
    const halfWidth = ((range.to - range.from) / 2) * multiplier;
    timeScale.setVisibleLogicalRange({ from: center - halfWidth, to: center + halfWidth });
  }

  function resetChart() {
    chartRef.current?.timeScale?.().resetTimeScale?.();
    chartRef.current?.timeScale?.().fitContent();
  }

  function toggleFullscreen() {
    if (!shellRef.current) {
      return;
    }
    if (document.fullscreenElement) {
      document.exitFullscreen?.();
      return;
    }
    shellRef.current.requestFullscreen?.();
  }

  async function downloadSnapshot() {
    const snapshot = chartRef.current?.takeScreenshot?.();
    if (!snapshot) {
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = snapshot.width * 2;
    canvas.height = snapshot.height * 2;
    const context = canvas.getContext("2d");
    context.scale(2, 2);
    context.drawImage(snapshot, 0, 0);
    const svg = shellRef.current?.querySelector(".drawing-layer");
    if (svg) {
      const serialized = new XMLSerializer().serializeToString(svg);
      const image = new Image();
      image.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(serialized)}`;
      await new Promise((resolve) => {
        image.onload = resolve;
        image.onerror = resolve;
      });
      context.drawImage(image, 0, 0, snapshot.width, snapshot.height);
    }
    const link = document.createElement("a");
    link.download = `${String(symbol || "chart").replace(/\s+/g, "-").toLowerCase()}-${rangeKey}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }

  function drawingPointFromEvent(event) {
    const box = hostRef.current?.getBoundingClientRect();
    if (!box) {
      return { x: 0.5, y: 0.5 };
    }
    return {
      x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
      y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height)),
    };
  }

  function denormalizePoint(point) {
    const box = hostRef.current?.getBoundingClientRect();
    const width = box?.width || 1;
    const height = box?.height || 1;
    return { x: point.x * width, y: point.y * height };
  }

  function createDrawing(tool, first, second) {
    const label = tool === "text" ? window.prompt("Text", "Note") || "Text" : "";
    return {
      id: stableId("draw"),
      type: tool,
      points: [first, second || { x: Math.min(1, first.x + 0.18), y: Math.min(1, first.y + 0.12) }],
      text: label,
      color: "#adc6ff",
    };
  }

  function handleDrawingPointerDown(event) {
    if (!activeDrawingTool) {
      return;
    }
    event.preventDefault();
    const point = drawingPointFromEvent(event);
    if (!drawingDraftRef.current) {
      drawingDraftRef.current = point;
      if (["horizontal", "vertical", "text"].includes(activeDrawingTool)) {
        const nextDrawing = createDrawing(activeDrawingTool, point);
        onDrawingsChange?.([...(drawings || []), nextDrawing]);
        onSelectedDrawingIdChange?.(nextDrawing.id);
        drawingDraftRef.current = null;
        onActiveDrawingToolChange?.("");
      }
      return;
    }
    const nextDrawing = createDrawing(activeDrawingTool, drawingDraftRef.current, point);
    onDrawingsChange?.([...(drawings || []), nextDrawing]);
    onSelectedDrawingIdChange?.(nextDrawing.id);
    drawingDraftRef.current = null;
    onActiveDrawingToolChange?.("");
  }

  function moveDrawing(drawing, delta) {
    return {
      ...drawing,
      points: drawing.points.map((point) => ({
        x: Math.max(0, Math.min(1, point.x + delta.x)),
        y: Math.max(0, Math.min(1, point.y + delta.y)),
      })),
    };
  }

  function beginDrawingDrag(event, drawing) {
    event.stopPropagation();
    onSelectedDrawingIdChange?.(drawing.id);
    dragDrawingRef.current = { id: drawing.id, start: drawingPointFromEvent(event), original: drawing };
  }

  function handleDrawingMove(event) {
    if (!dragDrawingRef.current) {
      return;
    }
    const point = drawingPointFromEvent(event);
    const delta = {
      x: point.x - dragDrawingRef.current.start.x,
      y: point.y - dragDrawingRef.current.start.y,
    };
    onDrawingsChange?.((drawings || []).map((item) => (
      item.id === dragDrawingRef.current.id ? moveDrawing(dragDrawingRef.current.original, delta) : item
    )));
  }

  function endDrawingDrag() {
    dragDrawingRef.current = null;
  }

  function deleteSelectedDrawing() {
    if (!selectedDrawingId) {
      return;
    }
    onDrawingsChange?.((drawings || []).filter((item) => item.id !== selectedDrawingId));
    onSelectedDrawingIdChange?.("");
  }

  function togglePanel(panel) {
    setOpenPanel((current) => (current === panel ? "" : panel));
  }

  function addIndicator() {
    const type = String(indicatorType || "EMA").toUpperCase();
    const next = {
      id: stableId("indicator"),
      type,
      period: Math.max(1, Number(indicatorPeriod) || 20),
      multiplier: type === "BB" ? 2 : (type === "SUPERTREND" ? 3 : undefined),
      color: ["#f7c948", "#7dd3fc", "#f472b6", "#40e56c", "#ff8f70"][(indicators || []).length % 5],
      enabled: true,
    };
    onIndicatorsChange?.([...(indicators || []), next]);
  }

  function startReplay() {
    const targetTime = replayStart ? Math.floor(new Date(replayStart).getTime() / 1000) : displayCandles[0]?.time;
    const index = Math.max(0, candles.findIndex((row) => row.time >= targetTime));
    onReplayChange?.({ active: true, playing: true, speed: replay?.speed || 1, index: index >= 0 ? index : 0 });
  }

  function createPriceAlert() {
    const price = Number(alertPrice);
    if (!Number.isFinite(price)) {
      return;
    }
    onCreateAlert?.({ id: stableId("alert"), symbol, type: "price", operator: alertOperator, value: price, enabled: true });
    setAlertPrice("");
  }

  function renderDrawing(drawing) {
    const [aRaw, bRaw] = drawing.points || [];
    if (!aRaw) {
      return null;
    }
    const a = denormalizePoint(aRaw);
    const b = denormalizePoint(bRaw || aRaw);
    const selected = drawing.id === selectedDrawingId;
    const stroke = selected ? "#f7c948" : (drawing.color || "#adc6ff");
    const common = {
      key: drawing.id,
      className: `drawing-shape ${selected ? "selected" : ""}`,
      onPointerDown: (event) => beginDrawingDrag(event, drawing),
    };
    if (drawing.type === "horizontal") {
      return <line {...common} x1="0" x2="100%" y1={a.y} y2={a.y} stroke={stroke} strokeWidth="2" />;
    }
    if (drawing.type === "vertical") {
      return <line {...common} x1={a.x} x2={a.x} y1="0" y2="100%" stroke={stroke} strokeWidth="2" />;
    }
    if (drawing.type === "rectangle") {
      return <rect {...common} x={Math.min(a.x, b.x)} y={Math.min(a.y, b.y)} width={Math.abs(b.x - a.x)} height={Math.abs(b.y - a.y)} fill="rgba(173,198,255,0.08)" stroke={stroke} strokeWidth="2" />;
    }
    if (drawing.type === "circle") {
      const radius = Math.max(8, Math.hypot(b.x - a.x, b.y - a.y));
      return <circle {...common} cx={a.x} cy={a.y} r={radius} fill="rgba(173,198,255,0.06)" stroke={stroke} strokeWidth="2" />;
    }
    if (drawing.type === "text") {
      return <text {...common} x={a.x} y={a.y} fill={stroke} fontSize="14" fontWeight="700">{drawing.text || "Text"}</text>;
    }
    if (drawing.type === "fib") {
      const levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
      return (
        <g {...common}>
          {levels.map((level) => {
            const y = a.y + ((b.y - a.y) * level);
            return <line key={level} x1={Math.min(a.x, b.x)} x2={Math.max(a.x, b.x)} y1={y} y2={y} stroke={stroke} strokeWidth="1.5" opacity="0.85" />;
          })}
        </g>
      );
    }
    if (drawing.type === "arrow") {
      return <line {...common} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={stroke} strokeWidth="2.4" markerEnd="url(#drawing-arrow)" />;
    }
    const x2 = drawing.type === "ray" ? b.x + ((b.x - a.x) * 4) : b.x;
    const y2 = drawing.type === "ray" ? b.y + ((b.y - a.y) * 4) : b.y;
    return <line {...common} x1={a.x} y1={a.y} x2={x2} y2={y2} stroke={stroke} strokeWidth="2.2" />;
  }

  function renderDockPanel() {
    if (openPanel === "view") {
      return (
        <div className="chart-dock-panel">
          <div className="view-panel">
            <button
              type="button"
              className={`line-button ${chartType === "candles" ? "active" : ""}`}
              onClick={() => setChartType("candles")}
            >
              Candles
            </button>
            <button
              type="button"
              className={`line-button ${chartType === "bars" ? "active" : ""}`}
              onClick={() => setChartType("bars")}
            >
              Bars
            </button>
            <button type="button" className={`line-button ${showVolume ? "active" : ""}`} onClick={() => setShowVolume((current) => !current)}>Volume</button>
            <button type="button" className={`line-button ${showEma ? "active" : ""}`} onClick={() => setShowEma((current) => !current)}>EMA 9/21</button>
            <button type="button" className={`line-button ${showVwap ? "active" : ""}`} onClick={() => setShowVwap((current) => !current)}>VWAP</button>
            <button type="button" className={`line-button ${crosshairMode === "crosshair" ? "active" : ""}`} onClick={() => setCrosshairMode((current) => (current === "crosshair" ? "cursor" : "crosshair"))}>Crosshair</button>
            <button type="button" className="tool-button" onClick={() => zoomChart(0.72)} title="Zoom in">
              <span className="material-symbols-outlined" aria-hidden="true">zoom_in</span>
            </button>
            <button type="button" className="tool-button" onClick={() => zoomChart(1.28)} title="Zoom out">
              <span className="material-symbols-outlined" aria-hidden="true">zoom_out</span>
            </button>
            <button type="button" className="tool-button" onClick={resetChart} title="Reset chart">
              <span className="material-symbols-outlined" aria-hidden="true">restart_alt</span>
            </button>
            <button type="button" className="tool-button" onClick={goLive} title="Go to live candle">
              <span className="material-symbols-outlined" aria-hidden="true">my_location</span>
            </button>
          </div>
        </div>
      );
    }
    if (openPanel === "drawings") {
      return (
        <div className="chart-dock-panel">
          <div className="drawing-tool-strip" aria-label="Drawing tools">
            {DRAWING_TOOLS.map((tool) => (
              <button
                key={tool.key}
                type="button"
                className={`tool-button ${activeDrawingTool === tool.key ? "active" : ""}`}
                onClick={() => onActiveDrawingToolChange?.(activeDrawingTool === tool.key ? "" : tool.key)}
                title={tool.label}
              >
                <span className="material-symbols-outlined" aria-hidden="true">{tool.icon}</span>
              </button>
            ))}
            <button type="button" className="tool-button" onClick={deleteSelectedDrawing} title="Delete drawing">
              <span className="material-symbols-outlined" aria-hidden="true">delete</span>
            </button>
            {activeDrawingTool ? <span className="dock-note">Click chart to place {activeDrawingTool}</span> : null}
          </div>
        </div>
      );
    }
    if (openPanel === "indicators") {
      return (
        <div className="chart-dock-panel">
          <div className="indicator-bar">
            <select className="mini-select" value={indicatorType} onChange={(event) => setIndicatorType(event.target.value)}>
              {["EMA", "SMA", "VWAP", "BB", "ATR", "SUPERTREND"].map((item) => <option key={item} value={item}>{item}</option>)}
            </select>
            <input className="mini-input" type="number" min="1" max="300" value={indicatorPeriod} onChange={(event) => setIndicatorPeriod(event.target.value)} />
            <button type="button" className="line-button" onClick={addIndicator}>Add</button>
            {(indicators || []).map((indicator) => (
              <button
                key={indicator.id}
                type="button"
                className={`indicator-pill ${indicator.enabled ? "active" : ""}`}
                onClick={() => onIndicatorsChange?.((indicators || []).map((item) => item.id === indicator.id ? { ...item, enabled: !item.enabled } : item))}
                onDoubleClick={() => onIndicatorsChange?.((indicators || []).filter((item) => item.id !== indicator.id))}
                title="Click to toggle, double-click to remove"
              >
                {indicatorLabel(indicator)}
              </button>
            ))}
          </div>
        </div>
      );
    }
    if (openPanel === "replay") {
      return (
        <div className="chart-dock-panel">
          <div className="replay-panel">
            <input className="mini-input wide" type="datetime-local" value={replayStart} onChange={(event) => setReplayStart(event.target.value)} />
            <button type="button" className="line-button" onClick={startReplay}>Replay</button>
            <button type="button" className="line-button" disabled={!replay?.active} onClick={() => onReplayChange?.({ ...(replay || {}), playing: !replay?.playing })}>{replay?.playing ? "Pause" : "Play"}</button>
            <select className="mini-select" value={replay?.speed || 1} onChange={(event) => onReplayChange?.({ ...(replay || {}), speed: Number(event.target.value) })}>
              {REPLAY_SPEEDS.map((speed) => <option key={speed} value={speed}>{speed}x</option>)}
            </select>
            <button type="button" className="line-button" disabled={!replay?.active} onClick={() => onReplayChange?.({ active: false, playing: false, speed: 1, index: 0 })}>Stop</button>
          </div>
        </div>
      );
    }
    if (openPanel === "alerts") {
      return (
        <div className="chart-dock-panel">
          <div className="alert-panel">
            <select className="mini-select" value={alertOperator} onChange={(event) => setAlertOperator(event.target.value)}>
              <option value=">">Price &gt;</option>
              <option value="<">Price &lt;</option>
            </select>
            <input className="mini-input" type="number" value={alertPrice} onChange={(event) => setAlertPrice(event.target.value)} placeholder="Price" />
            <button type="button" className="line-button" onClick={createPriceAlert}>Alert</button>
            <span className="chip">{formatCount((alerts || []).filter((item) => item.enabled).length)} alerts</span>
          </div>
        </div>
      );
    }
    if (openPanel === "templates") {
      return (
        <div className="chart-dock-panel">
          <div className="template-panel">
            <input className="mini-input wide" type="text" value={templateName} onChange={(event) => setTemplateName(event.target.value)} placeholder="Template" />
            <button type="button" className="line-button" onClick={() => templateName && onSaveTemplate?.(templateName)}>Save</button>
            <button type="button" className="line-button" onClick={() => templateName && onApplyTemplate?.(templateName)}>Apply</button>
          </div>
        </div>
      );
    }
    return null;
  }

  return (
    <article ref={shellRef} className="panel chart-panel">
      <div className="panel-head">
        <div>
          <h2>{symbol}</h2>
          <p>{deferredChart?.instrument_key || "Live market workspace"}</p>
        </div>
        <div className="chart-meta">
          <span className="chip">
            {activeRangeLabel}
            {" / "}
            {activeIntervalLabel}
          </span>
          <span className={`chip ${deferredChart?.supports_live ? "emphasis" : ""}`}>
            {deferredChart?.supports_live ? "Live WS" : "Archive"}
          </span>
        </div>
      </div>
      <div className="chart-terminal-toolbar simple-chart-toolbar">
        <div className="chart-fixed-mode">
          <span className="chip emphasis">1m candles</span>
        </div>
        <div className="chart-tool-strip" aria-label="Chart tools">
          <button type="button" className="tool-button" onClick={fitChart} title="Fit chart">
            <span className="material-symbols-outlined" aria-hidden="true">fit_screen</span>
          </button>
          <button type="button" className="tool-button" onClick={downloadSnapshot} title="Download snapshot">
            <span className="material-symbols-outlined" aria-hidden="true">photo_camera</span>
          </button>
          <button type="button" className="tool-button" onClick={toggleFullscreen} title="Fullscreen">
            <span className="material-symbols-outlined" aria-hidden="true">fullscreen</span>
          </button>
        </div>
      </div>
      <div className="status-row chart-status-row">
        <span className="chip">From {formatDate(deferredChart?.start_date)}</span>
        <span className="chip">To {formatDate(deferredChart?.end_date)}</span>
        <span className="chip">Loaded {formatCount(displayCandles?.length)} candles</span>
        {deferredChart?.available_count ? <span className="chip">Available {formatCount(deferredChart.available_count)}</span> : null}
        {deferredChart?.available_count && displayCandles.length >= deferredChart.available_count
          ? <span className="chip emphasis">Full history loaded</span>
          : null}
        {(!deferredChart?.available_count || displayCandles.length < deferredChart.available_count)
          ? <span className="chip">Scroll left for older</span>
          : null}
      </div>
      <div className="chart-readout">
        <span>{formatChartCrosshairTime(activeCandle?.time) || "Latest"}</span>
        <strong>O {formatMoney(activeCandle?.open)}</strong>
        <strong>H {formatMoney(activeCandle?.high)}</strong>
        <strong>L {formatMoney(activeCandle?.low)}</strong>
        <strong>C <span className={activeTone}>{formatMoney(activeCandle?.close)}</span></strong>
        <strong>Vol {formatCount(activeCandle?.volume)}</strong>
        <strong className={activeTone}>{candleChangePct(activeCandle)}</strong>
      </div>
      <div className="chart-legend-row">
        <span className="legend-line price">Price</span>
        {showVolume ? <span className="legend-line volume">Volume</span> : null}
      </div>
      <div className="chart-stage">
        <div ref={hostRef} className="chart-canvas" />
      </div>
    </article>
  );
}

function LiveDataCard({ price, freshness, history, chart, stream }) {
  const intervals = history?.intervals || [];
  const minuteRow = intervals.find((row) => row.interval === "1minute") || {};
  const dayRow = intervals.find((row) => row.interval === "day") || {};
  const marketStatus = freshness?.market_status || "-";
  const runtime = stream?.runtime || {};
  const sourceRunning = Boolean(runtime.running);
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Live Feed</h2>
        </div>
        <span className={`tag ${marketStatus === "live" ? "buy" : "hold"}`}>{marketStatus}</span>
      </div>

      <div className="info-grid">
        <div className="info-tile">
          <span>Last</span>
          <strong>{formatMoney(price?.last)}</strong>
        </div>
        <div className="info-tile">
          <span>Change</span>
          <strong>{formatSignedMoney(price?.change)}</strong>
        </div>
        <div className="info-tile">
          <span>Candle Age</span>
          <strong>{formatAge(freshness?.latest_candle_age_seconds)}</strong>
        </div>
        <div className="info-tile">
          <span>Chart Range</span>
          <strong>{chart?.label || "-"}</strong>
        </div>
      </div>

      <div className="chip-row">
        <span className={`chip ${marketStatus === "live" ? "emphasis" : ""}`}>WS {marketStatus}</span>
        <span className={`chip ${sourceRunning ? "emphasis" : ""}`}>Source {sourceRunning ? "running" : "stopped"}</span>
        <span className="chip">1m {formatDateTime(minuteRow.latest_ts)}</span>
        <span className="chip">Day {formatDateTime(dayRow.latest_ts)}</span>
      </div>

      <div className="chip-row">
        <span className="chip">Exchange {formatDateTime(stream?.latest_exchange_ts)}</span>
        <span className="chip">Persist {formatNs(stream?.estimated_exchange_to_persist_latency_ns)}</span>
        <span className="chip">Now {formatNs(stream?.estimated_exchange_to_now_latency_ns)}</span>
      </div>

      {runtime?.last_error ? (
        <div className="stack-list">
          <div className="note-row">Last stream error: {runtime.last_error}</div>
        </div>
      ) : null}
    </article>
  );
}

function StrategySignalCard({ signal, option }) {
  const [showDetails, setShowDetails] = useState(false);
  const details = signal?.details || {};
  const optionSignal = option?.signal || {};
  const tone = signalTone(signal);
  const readiness = entryReadiness(signal, optionSignal);
  const checks = [
    {
      label: "EMA 9 > EMA 21 cross",
      value: signal?.action === "BUY" ? details.ema_cross_up : signal?.action === "SELL" ? details.ema_cross_down : null,
      meta: `EMA 9 ${formatMoney(details.ema_9)} / EMA 21 ${formatMoney(details.ema_21)}`,
    },
    {
      label: "RSI filter",
      value: signal?.action === "BUY" ? details.rsi_buy_ok : signal?.action === "SELL" ? details.rsi_sell_ok : null,
      meta: `RSI ${details.rsi_14 ?? "-"}`,
    },
    {
      label: "Signal candle direction",
      value: signal?.action === "BUY" ? details.bullish_candle : signal?.action === "SELL" ? details.bearish_candle : null,
      meta: `Open ${formatMoney(details.open)} / Close ${formatMoney(details.close)}`,
    },
    {
      label: "Above-average volume",
      value: details.volume_ok,
      meta: `Vol ${formatCount(details.volume)} / Avg ${formatCount(details.volume_sma_20)}`,
    },
    {
      label: "Previous candle breakout",
      value: signal?.action === "BUY" ? details.break_prev_high : signal?.action === "SELL" ? details.break_prev_low : null,
      meta: `Prev high ${formatMoney(details.prev_high)} / Prev low ${formatMoney(details.prev_low)}`,
    },
    {
      label: "Entry window open",
      value: details.entry_window_open,
      meta: `${details.entry_window_start || "09:45"}-${details.entry_window_end || "15:00"} IST / ${details.window_status || "-"}`,
    },
  ];

  return (
    <article className={`panel signal-panel ${tone}`}>
      <div className="panel-head">
        <div>
          <h2>Strategy Signal</h2>
        </div>
        <div className="panel-actions">
          <button
            type="button"
            className="line-button"
            onClick={() => setShowDetails((current) => !current)}
          >
            {showDetails ? "Hide Details" : "Details"}
          </button>
          <span className={`tag ${tone}`}>{signalLabel(signal)}</span>
        </div>
      </div>

      <div className="signal-hero">
        <div>
          <span>Direction</span>
          <strong className={tone}>{signalLabel(signal)}</strong>
        </div>
        <div>
          <span>Readiness</span>
          <strong className={readiness.tone}>{readiness.label}</strong>
        </div>
        <div>
          <span>Score</span>
          <strong>{signal?.score ?? "-"}</strong>
        </div>
        <div>
          <span>Confidence</span>
          <strong>{formatPct(Number(signal?.confidence || 0) * 100)}</strong>
        </div>
      </div>

      <div className="chip-row">
        <span className="chip">{details.strategy_interval || "-"}</span>
        <span className="chip">Option {details.option_preference || "ATM_ONLY"}</span>
        <span className="chip">CE score {details.score_buy ?? "-"}</span>
        <span className="chip">PE score {details.score_sell ?? "-"}</span>
      </div>

      {showDetails ? (
        <div className="signal-checklist">
          {checks.map((item) => (
            <div
              key={item.label}
              className={`condition-row ${item.value === true ? "on" : item.value === false ? "off" : "neutral"}`}
            >
              <strong>{item.label}</strong>
              <span>{item.value === null || item.value === undefined ? "-" : formatFlag(item.value)}</span>
              <small>{item.meta}</small>
            </div>
          ))}
        </div>
      ) : null}

      <div className="chip-row">
        <span className="chip">Option action {optionSignal.action || "-"}</span>
        <span className="chip">{optionSignal.option_type || "-"} {optionSignal.strike || "-"}</span>
        <span className="chip">Entry {formatMoney(optionSignal.entry_price)}</span>
        <span className="chip">SL {formatMoney(optionSignal.stop_loss)}</span>
        <span className="chip">TP {formatMoney(optionSignal.take_profit)}</span>
      </div>

    </article>
  );
}

function EngineStrikeCard({ signal, option }) {
  const optionSignal = option?.signal || {};
  const action = optionSignal.action || signal?.action || "HOLD";
  const tone = actionTone(action);
  const strike = optionSignal.strike || optionSignal.strike_price || "-";
  const optionType = optionSignal.option_type || "-";
  const readiness = entryReadiness(signal, optionSignal);
  const rr = riskReward(optionSignal.entry_price, optionSignal.stop_loss, optionSignal.take_profit);
  return (
    <article className={`panel engine-strike-card ${tone}`}>
      <div className="panel-head">
        <div>
          <h2>Engine Strike</h2>
        </div>
        <span className={`tag ${readiness.tone}`}>{readiness.label}</span>
      </div>

      <div className="engine-strike-main">
        <div>
          <span>Contract</span>
          <strong>{strike} {optionType}</strong>
        </div>
        <div>
          <span>Action</span>
          <strong>{action}</strong>
        </div>
      </div>

      <div className="info-grid">
        <div className="info-tile">
          <span>Entry</span>
          <strong>{formatMoney(optionSignal.entry_price)}</strong>
        </div>
        <div className="info-tile">
          <span>Stop Loss</span>
          <strong>{formatMoney(optionSignal.stop_loss)}</strong>
        </div>
        <div className="info-tile">
          <span>Target</span>
          <strong>{formatMoney(optionSignal.take_profit)}</strong>
        </div>
        <div className="info-tile">
          <span>Premium</span>
          <strong>{formatMoney(optionSignal.current_premium || optionSignal.entry_price)}</strong>
        </div>
        <div className="info-tile">
          <span>Risk / Reward</span>
          <strong>{rr}</strong>
        </div>
      </div>
    </article>
  );
}

function DataCoverageCard({ history }) {
  const records = history?.records || {};
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Data Status</h2>
        </div>
      </div>

      <div className="info-grid">
        <div className="info-tile">
          <span>Retention</span>
          <strong>{history?.retention_years || 2} years</strong>
        </div>
        <div className="info-tile">
          <span>Expected End</span>
          <strong>{formatDate(history?.expected_end_date)}</strong>
        </div>
        <div className="info-tile">
          <span>Option Quotes</span>
          <strong>{formatCount(records.option_quotes)}</strong>
        </div>
        <div className="info-tile">
          <span>Latest Quote</span>
          <strong>{formatDateTime(history?.latest_option_quote_ts)}</strong>
        </div>
      </div>
    </article>
  );
}

function SessionCard({ calendar, option, freshness }) {
  const upcoming = (calendar?.upcoming_days || []).slice(0, 6);
  const expiries = option?.available_expiries || [];
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Trading Session</h2>
        </div>
      </div>

      <div className="info-grid">
        <div className="info-tile">
          <span>Status</span>
          <strong>{calendar?.session_status || freshness?.market_status || "-"}</strong>
        </div>
        <div className="info-tile">
          <span>Session</span>
          <strong>{formatTime(calendar?.market_session?.start)} - {formatTime(calendar?.market_session?.end)}</strong>
        </div>
        <div className="info-tile">
          <span>Previous Day</span>
          <strong>{formatDate(calendar?.previous_trading_day)}</strong>
        </div>
        <div className="info-tile">
          <span>Next Day</span>
          <strong>{formatDate(calendar?.next_trading_day)}</strong>
        </div>
      </div>

      <div className="chip-row">
        {expiries.slice(0, 5).map((expiry) => (
          <span key={expiry} className="chip emphasis">{formatDate(expiry)}</span>
        ))}
      </div>

      <div className="stack-list">
        {upcoming.map((row) => (
          <div key={row.date} className="note-row compact">
            <strong>{row.label}</strong>
            <span>{row.is_trading_day ? "Trading day" : "Weekend / holiday"}</span>
            <span>{row.is_expiry ? "Expiry" : ""}</span>
          </div>
        ))}
      </div>
    </article>
  );
}

function TradingCalendar({ calendar }) {
  const month = calendar?.current_month || {};
  const blanks = Array.from({ length: Number(month.leading_blanks || 0) });
  const days = month.days || [];

  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>{month.label || "Trading calendar"}</h2>
          <p>Current month in IST with trading days and expiry markers.</p>
        </div>
      </div>

      <div className="calendar-legend">
        <span className="legend"><i className="legend-dot trading" /> Trading day</span>
        <span className="legend"><i className="legend-dot closed" /> Closed</span>
        <span className="legend"><i className="legend-dot expiry" /> Expiry</span>
        <span className="legend"><i className="legend-dot today" /> Today</span>
      </div>

      <div className="calendar-weekdays">
        {["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((item) => (
          <span key={item}>{item}</span>
        ))}
      </div>

      <div className="calendar-grid">
        {blanks.map((_, index) => (
          <div key={`blank-${index}`} className="calendar-cell blank" />
        ))}
        {days.map((day) => (
          <div
            key={day.date}
            className={`calendar-cell ${day.is_trading_day ? "trading" : "closed"} ${day.is_expiry ? "expiry" : ""} ${day.is_today ? "today" : ""}`}
          >
            <strong>{day.day}</strong>
            <small>{day.weekday}</small>
          </div>
        ))}
      </div>
    </article>
  );
}

function HistoryWindow({ history }) {
  const records = history?.records || {};
  return (
    <section className="database-stack">
      <section className="metrics-grid">
        <MetricCard
          label="Retention Window"
          value={`${history?.retention_years || 2} years`}
          meta={`Target start ${formatDate(history?.target_start_date)}`}
          tone="neutral"
        />
        <MetricCard
          label="Option Quotes"
          value={formatCount(records.option_quotes)}
          meta={`Latest quote ${formatDateTime(history?.latest_option_quote_ts)}`}
          tone="neutral"
        />
        <MetricCard
          label="Orders"
          value={formatCount(records.orders)}
          meta={`${formatCount(records.closed_trades)} closed trades in window`}
          tone="neutral"
        />
        <MetricCard
          label="Closed Trades"
          value={formatCount(records.closed_trades)}
          meta={`Today ${formatDate(history?.today_ist)}`}
          tone="neutral"
        />
      </section>

      <article className="panel">
        <div className="panel-head">
          <div>
            <h2>Data Window</h2>
            <p>Static database totals without interval coverage scans.</p>
          </div>
        </div>
        <div className="info-grid">
          <div className="info-tile"><span>Expected Start</span><strong>{formatDate(history?.expected_start_date)}</strong></div>
          <div className="info-tile"><span>Expected End</span><strong>{formatDate(history?.expected_end_date)}</strong></div>
          <div className="info-tile"><span>Signals</span><strong>{formatCount(records.signals)}</strong></div>
          <div className="info-tile"><span>Open Positions</span><strong>{formatCount(records.open_positions)}</strong></div>
        </div>
      </article>
    </section>
  );
}

function PositionsTable({ positions, onClose, onDelete, onInspect }) {
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Open Positions</h2>
          <p>Current premium, risk, and manual exit.</p>
        </div>
      </div>
      {!positions.length ? (
        <div className="empty-state">No open positions.</div>
      ) : (
        <div className="table-shell">
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Strike</th>
                <th>Type</th>
                <th>Entry</th>
                <th>Current</th>
                <th>P&amp;L</th>
                <th>SL</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((row) => (
                <tr key={row.position_id}>
                  <td>{row.symbol}</td>
                  <td>
                    <button
                      type="button"
                      className="inline-link"
                      onClick={() => onInspect({
                        symbol: row.symbol,
                        strike: row.strike,
                        optionType: row.option_type,
                        expiry: row.expiry,
                        positionId: row.position_id,
                      })}
                    >
                      {row.strike}
                    </button>
                  </td>
                  <td>{row.option_type}</td>
                  <td>{formatMoney(row.entry_premium)}</td>
                  <td>{formatMoney(row.current_premium)}</td>
                  <td className={Number(row.unrealized_pnl) >= 0 ? "positive" : "negative"}>
                    {formatSignedMoney(row.unrealized_pnl)}
                  </td>
                  <td>{formatMoney(row.current_sl)}</td>
                  <td>
                    <div className="button-row compact-actions">
                      <button type="button" className="line-button" onClick={() => onClose(row.position_id)}>Exit</button>
                      <button type="button" className="line-button" onClick={() => onDelete(row.position_id)}>Delete</button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  );
}

function TradesTable({ rows }) {
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Recent Trades</h2>
          <p>Last closed positions.</p>
        </div>
      </div>
      {!rows.length ? (
        <div className="empty-state">No closed trades yet.</div>
      ) : (
        <div className="table-shell">
          <table>
            <thead>
              <tr>
                <th>Entry time</th>
                <th>Strike</th>
                <th>Type</th>
                <th>Entry</th>
                <th>Exit</th>
                <th>P&amp;L</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.position_id}-${row.entry_time}`}>
                  <td>{formatDateTime(row.entry_time)}</td>
                  <td>{row.strike}</td>
                  <td>{row.option_type}</td>
                  <td>{formatMoney(row.entry_premium)}</td>
                  <td>{formatMoney(row.exit_premium)}</td>
                  <td className={Number(row.realized_pnl) >= 0 ? "positive" : "negative"}>
                    {formatSignedMoney(row.realized_pnl)}
                  </td>
                  <td>{row.exit_reason || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  );
}

function PortfolioCard({ mode, portfolio, onResetPaper, onRefresh, busy }) {
  const summary = portfolio?.summary || {};
  const positions = portfolio?.positions || [];
  const livePositions = portfolio?.positions || [];
  const funds = portfolio?.funds || {};
  const brokerErrors = portfolio?.errors || [];
  const rows = mode === "live" ? livePositions : positions;
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>{mode === "live" ? "Upstox Portfolio" : "Paper Portfolio"}</h2>
        </div>
        <div className="button-row">
          <button type="button" className="secondary-button" disabled={busy} onClick={onRefresh}>Refresh portfolio</button>
          {mode === "paper" ? (
            <button type="button" className="secondary-button" disabled={busy} onClick={onResetPaper}>Reset paper capital</button>
          ) : null}
        </div>
      </div>
      {mode === "paper" ? (
        <div className="info-grid">
          <div className="info-tile"><span>Available</span><strong>{formatMoney(summary.available_balance)}</strong></div>
          <div className="info-tile"><span>Invested</span><strong>{formatMoney(summary.invested_amount)}</strong></div>
          <div className="info-tile"><span>Equity</span><strong>{formatMoney(summary.equity)}</strong></div>
          <div className="info-tile"><span>Total P&amp;L</span><strong>{formatSignedMoney(summary.total_pnl)}</strong></div>
        </div>
      ) : (
        <div className="info-grid">
          <div className="info-tile"><span>Funds</span><strong>{formatMoney(funds.available_margin || funds.available_funds || 0)}</strong></div>
          <div className="info-tile"><span>Utilized</span><strong>{formatMoney(funds.utilised_margin || funds.used_margin || 0)}</strong></div>
          <div className="info-tile"><span>Positions</span><strong>{formatCount(rows.length)}</strong></div>
          <div className="info-tile"><span>Status</span><strong>{portfolio?.status || "-"}</strong></div>
        </div>
      )}
      {!rows.length ? (
        <div className="empty-state">{mode === "live" ? "No live broker positions." : "No paper positions."}</div>
      ) : (
        <div className="table-shell compact-table">
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Qty</th>
                <th>Avg</th>
                <th>Last</th>
                <th>P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 10).map((row, index) => (
                <tr key={`${row.symbol || row.tradingsymbol || "pos"}-${index}`}>
                  <td>{row.symbol || row.tradingsymbol || row.instrument_token || "-"}</td>
                  <td>{row.quantity || row.net_quantity || "-"}</td>
                  <td>{formatMoney(row.average_price || row.buy_price || row.entry_premium)}</td>
                  <td>{formatMoney(row.last_price || row.current_premium)}</td>
                  <td className={Number(row.pnl || row.unrealized_pnl || 0) >= 0 ? "positive" : "negative"}>
                    {formatSignedMoney(row.pnl || row.unrealized_pnl || 0)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {mode === "live" && brokerErrors.length ? (
        <div className="error-banner">
          Broker not ready: {parseBrokerError(brokerErrors[0].body) || parseBrokerError(brokerErrors[0]) || "Unable to fetch Upstox portfolio."}
        </div>
      ) : null}
    </article>
  );
}

function RiskControlsCard({ execution, stats }) {
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Risk Controls</h2>
        </div>
        <span className={`tag ${execution?.mode === "live" ? "sell" : "hold"}`}>
          {execution?.mode || "paper"}
        </span>
      </div>

      <div className="info-grid">
        <div className="info-tile">
          <span>Daily Loss Limit</span>
          <strong className="negative">{formatMoney(execution?.max_daily_loss_amount)}</strong>
        </div>
        <div className="info-tile">
          <span>Today P&amp;L</span>
          <strong className={Number(stats?.total_pnl_today) >= 0 ? "positive" : "negative"}>
            {formatSignedMoney(stats?.total_pnl_today)}
          </strong>
        </div>
        <div className="info-tile">
          <span>Open Risk</span>
          <strong>{formatSignedMoney(stats?.open_positions_unrealized_pnl || 0)}</strong>
        </div>
        <div className="info-tile">
          <span>Trades Today</span>
          <strong>{formatCount(stats?.total_trades_today || 0)}</strong>
        </div>
      </div>
    </article>
  );
}

const SETTINGS_PERCENT_FIELDS = new Set([
  "execution_per_trade_risk_pct",
  "execution_max_daily_loss_pct",
  "execution_stop_loss_pct",
  "tsl_activation_percent",
  "tsl_trail_percent",
  "target_profit_percent",
]);

function settingsToDraft(settings = {}) {
  const draft = {
    execution_enabled: Boolean(settings.execution_enabled),
    execution_symbols: Array.isArray(settings.execution_symbols) ? settings.execution_symbols : [],
    upstox_access_token: "",
    smtp_password: "",
  };
  [
    "execution_capital",
    "execution_per_trade_risk_pct",
    "execution_max_daily_loss_pct",
    "execution_max_simultaneous_trades",
    "execution_max_daily_trades",
    "execution_lot_size",
    "execution_premium_min",
    "execution_premium_max",
    "execution_stop_loss_pct",
    "tsl_activation_percent",
    "tsl_trail_percent",
    "target_profit_percent",
    "entry_window_start",
    "entry_window_end",
    "force_squareoff_time",
    "signal_min_score",
    "signal_cooldown_minutes",
    "smtp_enabled",
    "smtp_host",
    "smtp_port",
    "smtp_username",
    "smtp_from_email",
    "smtp_to_emails",
    "smtp_use_tls",
    "smtp_use_ssl",
  ].forEach((key) => {
    const value = settings[key];
    if (SETTINGS_PERCENT_FIELDS.has(key) && value !== null && value !== undefined && value !== "") {
      draft[key] = String(Number(value) * 100);
    } else {
      draft[key] = value === null || value === undefined ? "" : value;
    }
  });
  return draft;
}

function settingsDraftPayload(draft = {}) {
  const payload = {
    ...draft,
    execution_symbols: Array.isArray(draft.execution_symbols) ? draft.execution_symbols : [],
  };
  [
    "execution_capital",
    "execution_per_trade_risk_pct",
    "execution_max_daily_loss_pct",
    "execution_premium_min",
    "execution_premium_max",
    "execution_stop_loss_pct",
    "tsl_activation_percent",
    "tsl_trail_percent",
    "target_profit_percent",
    "signal_min_score",
  ].forEach((key) => {
    const raw = payload[key];
    if (raw === "" || raw === null || raw === undefined) {
      delete payload[key];
      return;
    }
    const value = Number(raw);
    payload[key] = SETTINGS_PERCENT_FIELDS.has(key) ? value / 100 : value;
  });
  [
    "execution_max_simultaneous_trades",
    "execution_max_daily_trades",
    "execution_lot_size",
    "signal_cooldown_minutes",
    "smtp_port",
  ].forEach((key) => {
    const raw = payload[key];
    if (raw === "" || raw === null || raw === undefined) {
      delete payload[key];
      return;
    }
    payload[key] = Number.parseInt(raw, 10);
  });
  if (!String(payload.upstox_access_token || "").trim()) {
    delete payload.upstox_access_token;
  }
  if (!String(payload.smtp_password || "").trim()) {
    delete payload.smtp_password;
  }
  return payload;
}

function formatDefaultValue(defaults, key, kind = "text") {
  const value = defaults?.[key];
  if (value === null || value === undefined || value === "") {
    return "";
  }
  if (kind === "percent") {
    return `${(Number(value) * 100).toFixed(1)}%`;
  }
  if (kind === "money") {
    return formatMoney(value);
  }
  if (kind === "bool") {
    return value ? "On" : "Off";
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(", ") : "None";
  }
  return String(value);
}

function SettingsField({ label, defaultText = "", children }) {
  return (
    <label className="settings-field">
      <span>{label}</span>
      {children}
      {defaultText ? <small className="settings-default">Default {defaultText}</small> : null}
    </label>
  );
}

function settingsInputValue(draft, key) {
  const value = draft?.[key];
  return value === null || value === undefined ? "" : value;
}

function SettingsWindow({
  data,
  draft,
  loading,
  saving,
  testingSmtp,
  notice,
  onDraftChange,
  onToggleSymbol,
  onSave,
  onReload,
  onTestSmtp,
}) {
  if (loading && !draft) {
    return <InlineLoader label="Loading settings..." />;
  }
  const settings = data?.settings || {};
  const defaults = data?.defaults || {};
  const availableSymbols = Array.from(new Set([...(data?.available_symbols || []), ...((draft?.execution_symbols || []))]));
  const brokerStatus = data?.broker?.status || "-";
  const update = (key, value) => onDraftChange?.(key, value);

  return (
    <section className="settings-screen">
      <article className="panel settings-toolbar">
        <div className="panel-head">
          <div>
            <h2>Settings</h2>
          </div>
          <div className="button-row">
            <span className={`tag ${brokerStatus === "ok" || brokerStatus === "paper" ? "buy" : "sell"}`}>
              Broker {brokerStatus}
            </span>
            <button type="button" className="secondary-button" disabled={loading || saving} onClick={onReload}>
              <span className="material-symbols-outlined" aria-hidden="true">refresh</span>
              Reload
            </button>
            <button type="button" className="secondary-button" disabled={saving || !draft} onClick={onSave}>
              <span className="material-symbols-outlined" aria-hidden="true">save</span>
              Save
            </button>
          </div>
        </div>
        {notice ? <div className="settings-notice">{notice}</div> : null}
      </article>

      <div className="settings-grid">
        <article className="panel settings-card settings-wide">
          <div className="panel-head">
            <div>
              <h2>Trading</h2>
            </div>
            <label className="settings-switch">
              <input type="checkbox" checked={Boolean(draft?.execution_enabled)} onChange={(event) => update("execution_enabled", event.target.checked)} />
              <span>Enabled</span>
            </label>
          </div>
          <div className="settings-symbol-grid">
            {availableSymbols.map((item) => (
              <button
                key={item}
                type="button"
                className={`settings-symbol ${draft?.execution_symbols?.includes(item) ? "active" : ""}`}
                onClick={() => onToggleSymbol?.(item)}
              >
                {item}
              </button>
            ))}
          </div>
          <div className="settings-default-line">
            Default symbols: {formatDefaultValue(defaults, "execution_symbols") || "None"}
          </div>
          <div className="settings-form-grid">
            <SettingsField label="Capital" defaultText={formatDefaultValue(defaults, "execution_capital", "money")}>
              <input className="select" type="number" min="0" value={settingsInputValue(draft, "execution_capital")} onChange={(event) => update("execution_capital", event.target.value)} />
            </SettingsField>
            <SettingsField label="Risk / Trade %" defaultText={formatDefaultValue(defaults, "execution_per_trade_risk_pct", "percent")}>
              <input className="select" type="number" min="0" step="0.1" value={settingsInputValue(draft, "execution_per_trade_risk_pct")} onChange={(event) => update("execution_per_trade_risk_pct", event.target.value)} />
            </SettingsField>
            <SettingsField label="Daily Loss %" defaultText={formatDefaultValue(defaults, "execution_max_daily_loss_pct", "percent")}>
              <input className="select" type="number" min="0" step="0.1" value={settingsInputValue(draft, "execution_max_daily_loss_pct")} onChange={(event) => update("execution_max_daily_loss_pct", event.target.value)} />
            </SettingsField>
            <SettingsField label="Lots" defaultText={formatDefaultValue(defaults, "execution_lot_size")}>
              <input className="select" type="number" min="1" value={settingsInputValue(draft, "execution_lot_size")} onChange={(event) => update("execution_lot_size", event.target.value)} />
            </SettingsField>
            <SettingsField label="Open Trades" defaultText={formatDefaultValue(defaults, "execution_max_simultaneous_trades")}>
              <input className="select" type="number" min="1" value={settingsInputValue(draft, "execution_max_simultaneous_trades")} onChange={(event) => update("execution_max_simultaneous_trades", event.target.value)} />
            </SettingsField>
            <SettingsField label="Daily Trades" defaultText={formatDefaultValue(defaults, "execution_max_daily_trades")}>
              <input className="select" type="number" min="1" value={settingsInputValue(draft, "execution_max_daily_trades")} onChange={(event) => update("execution_max_daily_trades", event.target.value)} />
            </SettingsField>
          </div>
        </article>

        <article className="panel settings-card">
          <div className="panel-head">
            <div>
              <h2>Risk Plan</h2>
            </div>
          </div>
          <div className="settings-form-grid single">
            <SettingsField label="Stop Loss %" defaultText={formatDefaultValue(defaults, "execution_stop_loss_pct", "percent")}>
              <input className="select" type="number" min="0" step="0.1" value={settingsInputValue(draft, "execution_stop_loss_pct")} onChange={(event) => update("execution_stop_loss_pct", event.target.value)} />
            </SettingsField>
            <SettingsField label="Target %" defaultText={formatDefaultValue(defaults, "target_profit_percent", "percent")}>
              <input className="select" type="number" min="0" step="0.1" value={settingsInputValue(draft, "target_profit_percent")} onChange={(event) => update("target_profit_percent", event.target.value)} />
            </SettingsField>
            <SettingsField label="TSL Trigger %" defaultText={formatDefaultValue(defaults, "tsl_activation_percent", "percent")}>
              <input className="select" type="number" min="0" step="0.1" value={settingsInputValue(draft, "tsl_activation_percent")} onChange={(event) => update("tsl_activation_percent", event.target.value)} />
            </SettingsField>
            <SettingsField label="TSL Trail %" defaultText={formatDefaultValue(defaults, "tsl_trail_percent", "percent")}>
              <input className="select" type="number" min="0" step="0.1" value={settingsInputValue(draft, "tsl_trail_percent")} onChange={(event) => update("tsl_trail_percent", event.target.value)} />
            </SettingsField>
            <label className="settings-switch">
              <input type="checkbox" checked={Boolean(draft?.tsl_immediate)} onChange={(event) => update("tsl_immediate", event.target.checked)} />
              <span>Immediate TSL</span>
            </label>
          </div>
        </article>

        <article className="panel settings-card">
          <div className="panel-head">
            <div>
              <h2>Session</h2>
            </div>
          </div>
          <div className="settings-form-grid single">
            <SettingsField label="Entry Start" defaultText={formatDefaultValue(defaults, "entry_window_start")}>
              <input className="select" type="time" value={settingsInputValue(draft, "entry_window_start")} onChange={(event) => update("entry_window_start", event.target.value)} />
            </SettingsField>
            <SettingsField label="Entry End" defaultText={formatDefaultValue(defaults, "entry_window_end")}>
              <input className="select" type="time" value={settingsInputValue(draft, "entry_window_end")} onChange={(event) => update("entry_window_end", event.target.value)} />
            </SettingsField>
            <SettingsField label="Square Off" defaultText={formatDefaultValue(defaults, "force_squareoff_time")}>
              <input className="select" type="time" value={settingsInputValue(draft, "force_squareoff_time")} onChange={(event) => update("force_squareoff_time", event.target.value)} />
            </SettingsField>
            <SettingsField label="Signal Score" defaultText={formatDefaultValue(defaults, "signal_min_score")}>
              <input className="select" type="number" min="0" max="100" step="0.5" value={settingsInputValue(draft, "signal_min_score")} onChange={(event) => update("signal_min_score", event.target.value)} />
            </SettingsField>
            <SettingsField label="Cooldown Min" defaultText={formatDefaultValue(defaults, "signal_cooldown_minutes")}>
              <input className="select" type="number" min="0" value={settingsInputValue(draft, "signal_cooldown_minutes")} onChange={(event) => update("signal_cooldown_minutes", event.target.value)} />
            </SettingsField>
          </div>
        </article>

        <article className="panel settings-card settings-wide">
          <div className="panel-head">
            <div>
              <h2>Broker Token</h2>
            </div>
            <span className={`tag ${settings.upstox_token_present ? "buy" : "sell"}`}>
              {settings.upstox_token_present ? "Token saved" : "Token missing"}
            </span>
          </div>
          <SettingsField label="Daily Upstox Token">
            <textarea className="settings-textarea" value={settingsInputValue(draft, "upstox_access_token")} onChange={(event) => update("upstox_access_token", event.target.value)} placeholder={settings.upstox_token_masked || "Paste token"} />
          </SettingsField>
        </article>

        <article className="panel settings-card settings-wide">
          <div className="panel-head">
            <div>
              <h2>SMTP</h2>
            </div>
            <div className="button-row">
              <span className={`tag ${settings.smtp_ready ? "buy" : "hold"}`}>{settings.smtp_ready ? "Ready" : "Not ready"}</span>
              <button type="button" className="secondary-button" disabled={testingSmtp || saving || !draft} onClick={onTestSmtp}>
                <span className="material-symbols-outlined" aria-hidden="true">outgoing_mail</span>
                Test
              </button>
            </div>
          </div>
          <div className="settings-form-grid">
            <label className="settings-switch">
              <input type="checkbox" checked={Boolean(draft?.smtp_enabled)} onChange={(event) => update("smtp_enabled", event.target.checked)} />
              <span>SMTP Enabled</span>
            </label>
            <SettingsField label="Host" defaultText={formatDefaultValue(defaults, "smtp_host")}>
              <input className="select" value={settingsInputValue(draft, "smtp_host")} onChange={(event) => update("smtp_host", event.target.value)} />
            </SettingsField>
            <SettingsField label="Port" defaultText={formatDefaultValue(defaults, "smtp_port")}>
              <input className="select" type="number" min="1" value={settingsInputValue(draft, "smtp_port")} onChange={(event) => update("smtp_port", event.target.value)} />
            </SettingsField>
            <SettingsField label="Username" defaultText={formatDefaultValue(defaults, "smtp_username")}>
              <input className="select" value={settingsInputValue(draft, "smtp_username")} onChange={(event) => update("smtp_username", event.target.value)} />
            </SettingsField>
            <SettingsField label="Password">
              <input className="select" type="password" value={settingsInputValue(draft, "smtp_password")} onChange={(event) => update("smtp_password", event.target.value)} placeholder={settings.smtp_password_present ? "Saved" : ""} />
            </SettingsField>
            <SettingsField label="From Email" defaultText={formatDefaultValue(defaults, "smtp_from_email")}>
              <input className="select" value={settingsInputValue(draft, "smtp_from_email")} onChange={(event) => update("smtp_from_email", event.target.value)} />
            </SettingsField>
            <SettingsField label="Recipients" defaultText={formatDefaultValue(defaults, "smtp_to_emails")}>
              <input className="select" value={settingsInputValue(draft, "smtp_to_emails")} onChange={(event) => update("smtp_to_emails", event.target.value)} />
            </SettingsField>
            <label className="settings-switch">
              <input type="checkbox" checked={Boolean(draft?.smtp_use_tls)} onChange={(event) => update("smtp_use_tls", event.target.checked)} />
              <span>TLS</span>
            </label>
            <label className="settings-switch">
              <input type="checkbox" checked={Boolean(draft?.smtp_use_ssl)} onChange={(event) => update("smtp_use_ssl", event.target.checked)} />
              <span>SSL</span>
            </label>
          </div>
        </article>
      </div>
    </section>
  );
}

function StrategyPerformanceCard({ rows }) {
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Strategy Performance</h2>
          <p>Win rate, realized P&amp;L, and drawdown by strategy.</p>
        </div>
      </div>
      {!rows.length ? (
        <div className="empty-state">No closed trades yet.</div>
      ) : (
        <div className="table-shell compact-table">
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>P&amp;L</th>
                <th>Drawdown</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.strategy}>
                  <td>{row.strategy}</td>
                  <td>{row.trades}</td>
                  <td>{formatPct(row.win_rate)}</td>
                  <td className={Number(row.realized_pnl) >= 0 ? "positive" : "negative"}>{formatSignedMoney(row.realized_pnl)}</td>
                  <td>{formatPct(row.max_drawdown)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  );
}

function TradeHistoryDashboard({ historyData, strategyRows, filters, onFilterChange, loading }) {
  const rows = historyData?.rows || [];
  const summary = historyData?.summary || {};
  return (
    <section className="double-grid">
      <article className="panel">
        <div className="panel-head">
          <div>
            <h2>Trade History</h2>
            <p>Filter closed trades by date and strategy.</p>
          </div>
        </div>
        <div className="filter-row">
          <label className="field">
            <span>From</span>
            <input className="select" type="date" value={filters.dateFrom} onChange={(e) => onFilterChange("dateFrom", e.target.value)} />
          </label>
          <label className="field">
            <span>To</span>
            <input className="select" type="date" value={filters.dateTo} onChange={(e) => onFilterChange("dateTo", e.target.value)} />
          </label>
          <label className="field">
            <span>Strategy</span>
            <input className="select" type="text" value={filters.strategy} onChange={(e) => onFilterChange("strategy", e.target.value)} placeholder="fast_live_breakout" />
          </label>
        </div>
        <div className="info-grid">
          <div className="info-tile"><span>Trades</span><strong>{summary.trades || 0}</strong></div>
          <div className="info-tile"><span>Wins</span><strong>{summary.wins || 0}</strong></div>
          <div className="info-tile"><span>Losses</span><strong>{summary.losses || 0}</strong></div>
          <div className="info-tile"><span>P&amp;L</span><strong>{formatSignedMoney(summary.realized_pnl)}</strong></div>
        </div>
        {loading ? <InlineLoader label="Loading trade history..." /> : (
          !rows.length ? <div className="empty-state">No trades for the selected filter.</div> : (
            <div className="table-shell compact-table">
              <table>
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Strategy</th>
                    <th>Contract</th>
                    <th>Entry</th>
                    <th>Exit</th>
                    <th>P&amp;L</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={`${row.position_id}-${row.entry_time}`}>
                      <td>{formatDate(row.entry_time)}</td>
                      <td>{row.strategy_name}</td>
                      <td>{row.strike} {row.option_type}</td>
                      <td>{formatMoney(row.entry_premium)}</td>
                      <td>{formatMoney(row.exit_premium)}</td>
                      <td className={Number(row.realized_pnl) >= 0 ? "positive" : "negative"}>{formatSignedMoney(row.realized_pnl)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}
      </article>
      <StrategyPerformanceCard rows={strategyRows} />
    </section>
  );
}

function ContractChartModal({ contract, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!contract) {
      return;
    }
    let active = true;
    async function load() {
      setLoading(true);
      setError("");
      try {
        const params = new URLSearchParams({
          symbol: contract.symbol,
          expiry: contract.expiry,
          strike: String(contract.strike),
          option_type: contract.optionType,
        });
        if (contract.positionId) {
          params.append("position_id", String(contract.positionId));
        }
        const payload = await apiFetch(`/api/live/option-contract-chart?${params.toString()}`);
        if (active) {
          setData(payload);
        }
      } catch (loadError) {
        if (active) {
          setError(loadError.message || "Unable to load strike chart.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }
    load();
    return () => {
      active = false;
    };
  }, [contract]);

  if (!contract) {
    return null;
  }

  const points = data?.points || [];
  const ltps = points.map((item) => Number(item.ltp)).filter((value) => Number.isFinite(value));
  const pnls = points.map((item) => Number(item.pnl)).filter((value) => Number.isFinite(value));
  const min = Math.min(...(ltps.length ? ltps : [0]));
  const max = Math.max(...(ltps.length ? ltps : [1]));
  const pnlMin = Math.min(...(pnls.length ? pnls : [0]));
  const pnlMax = Math.max(...(pnls.length ? pnls : [1]));
  const linePath = points.map((point, index) => {
    const x = points.length <= 1 ? 10 : (index / (points.length - 1)) * 560 + 10;
    const y = 180 - (((Number(point.ltp) - min) / ((max - min) || 1)) * 150);
    return `${index === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");
  const pnlPath = points.map((point, index) => {
    const x = points.length <= 1 ? 10 : (index / (points.length - 1)) * 560 + 10;
    const y = 180 - (((Number(point.pnl || 0) - pnlMin) / ((pnlMax - pnlMin) || 1)) * 150);
    return `${index === 0 ? "M" : "L"} ${x} ${y}`;
  }).join(" ");

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="chart-modal" onClick={(event) => event.stopPropagation()}>
        <div className="panel-head">
          <div>
            <h2>{contract.symbol} {contract.strike} {contract.optionType}</h2>
            <p>{formatDate(contract.expiry)}</p>
          </div>
          <button type="button" className="line-button" onClick={onClose}>Close</button>
        </div>
        {loading ? <InlineLoader label="Loading strike chart..." /> : null}
        {error ? <div className="error-banner">{error}</div> : null}
        {!loading && !error ? (
          <>
            <div className="info-grid">
              <div className="info-tile"><span>Entry</span><strong>{formatMoney(data?.entry_price)}</strong></div>
              <div className="info-tile"><span>Last</span><strong>{formatMoney(points[points.length - 1]?.ltp)}</strong></div>
              <div className="info-tile"><span>P&amp;L</span><strong>{formatSignedMoney(points[points.length - 1]?.pnl)}</strong></div>
              <div className="info-tile"><span>Points</span><strong>{formatCount(points.length)}</strong></div>
            </div>
            <div className="mini-chart-grid">
              <div>
                <div className="mini-chart-title">LTP</div>
                <svg viewBox="0 0 580 200" className="mini-chart">
                  <path d={linePath} fill="none" stroke="#adc6ff" strokeWidth="3" />
                </svg>
              </div>
              <div>
                <div className="mini-chart-title">P&amp;L</div>
                <svg viewBox="0 0 580 200" className="mini-chart">
                  <path d={pnlPath} fill="none" stroke="#40e56c" strokeWidth="3" />
                </svg>
              </div>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

function App() {
  const [layout, setLayout] = useState(() => mergeLayout(readStoredLayout()));
  const [theme, setTheme] = useState(() => mergeLayout(readStoredLayout()).theme || getInitialTheme());
  const [symbols, setSymbols] = useState([]);
  const [symbol, setSymbol] = useState(() => mergeLayout(readStoredLayout()).selectedSymbol || "");
  const [snapshot, setSnapshot] = useState(null);
  const [chart, setChart] = useState(null);
  const [portfolio, setPortfolio] = useState(null);
  const [tradeHistory, setTradeHistory] = useState(null);
  const [strategyPerformance, setStrategyPerformance] = useState([]);
  const [contractModal, setContractModal] = useState(null);
  const [chartRange, setChartRange] = useState("all");
  const [chartInterval, setChartInterval] = useState("1minute");
  const [chartLoading, setChartLoading] = useState(false);
  const [chartHistoryLoading, setChartHistoryLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [portfolioLoading, setPortfolioLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [error, setError] = useState("");
  const [streamState, setStreamState] = useState("connecting");
  const [busy, setBusy] = useState(false);
  const [runtimeMode, setRuntimeMode] = useState("paper");
  const [brokerHealth, setBrokerHealth] = useState(null);
  const [bootstrapped, setBootstrapped] = useState(false);
  const [activeView, setActiveView] = useState("overview");
  const [historyFilters, setHistoryFilters] = useState({ dateFrom: "", dateTo: "", strategy: "" });
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeDrawingTool, setActiveDrawingTool] = useState("");
  const [selectedDrawingId, setSelectedDrawingId] = useState("");
  const [replay, setReplay] = useState({ active: false, playing: false, speed: 1, index: 0 });
  const [alertEvents, setAlertEvents] = useState([]);
  const [settingsData, setSettingsData] = useState(null);
  const [settingsDraft, setSettingsDraft] = useState(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [smtpTesting, setSmtpTesting] = useState(false);
  const [settingsNotice, setSettingsNotice] = useState("");
  const chartCacheRef = useRef(new Map());
  const chartWarmRef = useRef(new Set());
  const chartHistoryWarmRef = useRef(new Set());
  const chartSelectionRef = useRef({ range: chartRange, interval: chartInterval });
  const reconnectTimerRef = useRef(null);
  const reconnectAttemptRef = useRef(0);
  const currentSymbolState = getSymbolState(layout, symbol);

  useEffect(() => {
    chartSelectionRef.current = { range: chartRange, interval: chartInterval };
  }, [chartRange, chartInterval]);

  useEffect(() => {
    writeStoredLayout(layout);
  }, [layout]);

  useEffect(() => {
    if (!symbol) {
      return;
    }
    setLayout((current) => upsertSymbolState({
      ...current,
      selectedSymbol: symbol,
      range: "all",
      timeframe: "1minute",
      theme,
    }, symbol, { range: "all", interval: "1minute" }));
  }, [symbol, chartRange, chartInterval, theme]);

  useEffect(() => {
    if (!symbol) {
      return;
    }
    setChartRange("all");
    setChartInterval("1minute");
  }, [symbol]);

  useEffect(() => {
    const onKeyDown = (event) => {
      const key = event.key.toLowerCase();
      if (event.ctrlKey && key === "k") {
        event.preventDefault();
        setSearchOpen(true);
        return;
      }
      if (key === "f") {
        event.preventDefault();
        window.dispatchEvent(new CustomEvent("alpha-fit-chart"));
        return;
      }
      if (key === "r") {
        event.preventDefault();
        window.dispatchEvent(new CustomEvent("alpha-reset-chart"));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch (_error) {
      // Ignore storage failures; the in-memory theme still applies.
    }
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    async function loadExecutionMode() {
      try {
        const payload = await apiFetch("/execution/mode");
        if (cancelled) {
          return;
        }
        setRuntimeMode(payload.mode || "paper");
        setBrokerHealth(payload.broker || null);
      } catch (_loadError) {
        if (!cancelled) {
          setRuntimeMode("paper");
          setBrokerHealth(null);
        }
      }
    }
    async function loadSymbols() {
      try {
        const payload = await apiFetch("/api/symbols");
        if (cancelled) {
          return;
        }
        const nextSymbols = Array.isArray(payload.symbols) && payload.symbols.length
          ? payload.symbols
          : ["Nifty 50"];
        setSymbols(nextSymbols);
        setSymbol((current) => current || nextSymbols[0]);
      } catch (loadError) {
        if (!cancelled) {
          setSymbols(["Nifty 50"]);
          setSymbol("Nifty 50");
          setError(loadError.message || "Unable to load symbols.");
        }
      }
    }
    loadExecutionMode();
    loadSymbols();
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedSnapshot = snapshot || {};
  const selectedChart = chart || selectedSnapshot.chart || {};

  useEffect(() => {
    if (!symbol) {
      return undefined;
    }
    let active = true;
    let socket;

    function scheduleReconnect() {
      if (!active) {
        return;
      }
      const base = 1000;
      const max = 10000;
      const delay = Math.min(max, base * (2 ** reconnectAttemptRef.current));
      reconnectAttemptRef.current += 1;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      reconnectTimerRef.current = setTimeout(() => {
        loadInitial();
      }, delay);
    }

    async function loadInitial() {
      try {
        if (!snapshot) {
          setLoading(true);
        }
        const data = await apiFetch(`/api/live/snapshot?symbol=${encodeURIComponent(symbol)}&include_static=false&include_chart=false&include_option=false`);
        if (!active) {
          return;
        }
        reconnectAttemptRef.current = 0;
        if (data.execution?.mode) {
          setRuntimeMode(data.execution.mode);
        }
        if (data.execution?.broker) {
          setBrokerHealth(data.execution.broker);
        }
        startTransition(() => {
          setSnapshot((current) => ({
            ...data,
            execution: {
              ...(data.execution || {}),
              broker: data.execution?.broker || current?.execution?.broker,
            },
          }));
          if (data?.chart?.range) {
            chartCacheRef.current.set(chartCacheKey(symbol, data.chart.range, data.chart.interval), data.chart);
            const selected = chartSelectionRef.current;
            if (selected.range === data.chart.range && selected.interval === data.chart.interval) {
              setChart(data.chart);
            }
          }
        });
        setBootstrapped(true);
        setError("");
      } catch (loadError) {
        if (active) {
          setError(loadError.message || "Unable to load snapshot.");
          scheduleReconnect();
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }

      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${protocol}://${window.location.host}/api/live/ws?symbol=${encodeURIComponent(symbol)}`);
      socket.addEventListener("open", () => {
        if (active) {
          setStreamState("live");
          setError("");
          reconnectAttemptRef.current = 0;
        }
      });
      socket.addEventListener("message", (event) => {
        if (!active) {
          return;
        }
        try {
          const message = JSON.parse(event.data);
          if (message.type === "snapshot") {
            if (message.payload?.execution?.mode) {
              setRuntimeMode(message.payload.execution.mode);
            }
            if (message.payload?.execution?.broker) {
              setBrokerHealth(message.payload.execution.broker);
            }
            startTransition(() => setSnapshot((current) => ({
              ...(current || {}),
              ...(message.payload || {}),
              execution: {
                ...(message.payload?.execution || current?.execution || {}),
                broker: message.payload?.execution?.broker || current?.execution?.broker,
              },
              notifications: message.payload?.notifications || current?.notifications || {},
              calendar: message.payload?.calendar || current?.calendar || {},
              history: message.payload?.history || current?.history || {},
            })));
            setStreamState("live");
            return;
          }
          if (message.type === "price") {
            startTransition(() => {
              setSnapshot((current) => mergeQuickUpdate(current, message.payload));
              if (chartSelectionRef.current.interval === "1minute") {
                setChart((current) => {
                  const next = mergeLiveChart(current, message.payload);
                  if (next?.range && next?.interval) {
                    chartCacheRef.current.set(chartCacheKey(symbol, next.range, next.interval), next);
                  }
                  return next;
                });
              }
            });
            setStreamState("live");
            return;
          }
          if (message.type === "error") {
            setError(message.payload?.detail || "Live stream error.");
          }
          setStreamState("live");
        } catch (parseError) {
          setError(parseError.message || "Invalid stream payload.");
        }
      });
      socket.addEventListener("close", () => {
        if (active) {
          setStreamState("reconnecting");
          scheduleReconnect();
        }
      });
      socket.addEventListener("error", () => {
        if (active) {
          setStreamState("reconnecting");
          scheduleReconnect();
        }
      });
    }

    loadInitial();
    return () => {
      active = false;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (socket) {
        socket.close();
      }
    };
  }, [symbol]);

  useEffect(() => {
    if (!symbol) {
      return undefined;
    }
    let active = true;

    async function loadChart() {
      try {
        const cacheKey = chartCacheKey(symbol, chartRange, chartInterval);
        const hasVisibleChart = Boolean(chartCacheRef.current.get(cacheKey) || chart || snapshot?.chart);
        setChartLoading(!hasVisibleChart);
        const data = await fetchChartPayload(symbol, chartRange, chartInterval);
        if (!active) {
          return;
        }
        chartCacheRef.current.set(cacheKey, data);
        setChart(data);
        setError("");
      } catch (loadError) {
        if (active) {
          setError(loadError.message || "Unable to load chart range.");
        }
      } finally {
        if (active) {
          setChartLoading(false);
        }
      }
    }

    const cached = chartCacheRef.current.get(chartCacheKey(symbol, chartRange, chartInterval));
    if (cached) {
      setChart(cached);
      setChartLoading(false);
    }

    loadChart();
    return () => {
      active = false;
    };
  }, [symbol, chartRange, chartInterval]);

  const stats = selectedSnapshot.stats || {};
  const price = selectedSnapshot.price || {};
  const freshness = selectedSnapshot.freshness || {};
  const option = selectedSnapshot.option || {};
  const signal = selectedSnapshot.signal || {};
  const execution = {
    ...(selectedSnapshot.execution || {}),
    mode: runtimeMode || selectedSnapshot.execution?.mode || "paper",
    broker: brokerHealth || selectedSnapshot.execution?.broker,
  };
  const brokerStatus = execution.broker?.status || (execution.mode === "live" ? "unknown" : "paper");
  const calendar = selectedSnapshot.calendar || {};
  const history = selectedSnapshot.history || {};
  const stream = selectedSnapshot.stream || {};
  const deferredPositions = useDeferredValue(selectedSnapshot.positions || []);
  const pnlTone = Number(stats.total_pnl_today) >= 0 ? "positive" : "negative";
  const priceTone = Number(price.change) >= 0 ? "positive" : "negative";

  const activeMeta = NAV_ITEMS.find((item) => item.id === activeView) || NAV_ITEMS[0];
  const summaryLine = selectedSnapshot.generated_at
    ? `${formatDateTime(selectedSnapshot.generated_at)} | candle age ${formatAge(freshness.latest_candle_age_seconds)} | ${freshness.market_status || "unknown"}`
    : "Waiting for first snapshot.";
  const searchNeedle = searchQuery.trim().toLowerCase();
  const searchResults = symbols
    .filter((item) => !searchNeedle || item.toLowerCase().includes(searchNeedle) || item.replace(/\s+/g, "").toLowerCase().includes(searchNeedle.replace(/\s+/g, "")))
    .slice(0, 20);

  async function refreshSnapshot() {
    const data = await apiFetch(`/api/live/snapshot?symbol=${encodeURIComponent(symbol)}&include_static=false&include_chart=false&include_option=false`);
    if (data.execution?.mode) {
      setRuntimeMode(data.execution.mode);
    }
    if (data.execution?.broker) {
      setBrokerHealth(data.execution.broker);
    }
    startTransition(() => setSnapshot((current) => ({
      ...data,
      execution: {
        ...(data.execution || {}),
        broker: data.execution?.broker || current?.execution?.broker,
      },
    })));
  }

  async function refreshPortfolio() {
    setPortfolioLoading(true);
    try {
      const data = await apiFetch("/execution/portfolio");
      startTransition(() => setPortfolio(data));
    } catch (loadError) {
      setError(loadError.message || "Unable to load portfolio.");
    } finally {
      setPortfolioLoading(false);
    }
  }

  async function refreshTradeHistory(nextFilters = historyFilters) {
    setHistoryLoading(true);
    try {
      const params = new URLSearchParams();
      if (nextFilters.dateFrom) params.append("date_from", nextFilters.dateFrom);
      if (nextFilters.dateTo) params.append("date_to", nextFilters.dateTo);
      if (nextFilters.strategy) params.append("strategy", nextFilters.strategy);
      const [historyPayload, strategyPayload] = await Promise.all([
        apiFetch(`/execution/trade-history?${params.toString()}`),
        apiFetch("/execution/strategy-performance"),
      ]);
      startTransition(() => {
        setTradeHistory(historyPayload);
        setStrategyPerformance(strategyPayload.rows || []);
      });
    } catch (loadError) {
      setError(loadError.message || "Unable to load trade history.");
    } finally {
      setHistoryLoading(false);
    }
  }

  function applySettingsPayload(data) {
    setSettingsData(data);
    setSettingsDraft(settingsToDraft(data?.settings || {}));
    if (data?.mode) {
      setRuntimeMode(data.mode);
    }
    if (data?.broker) {
      setBrokerHealth(data.broker);
    }
    if (Array.isArray(data?.available_symbols) && data.available_symbols.length) {
      setSymbols(data.available_symbols);
    }
  }

  async function loadRuntimeSettings(showLoader = true) {
    if (showLoader) {
      setSettingsLoading(true);
    }
    try {
      const data = await apiFetch("/execution/settings");
      applySettingsPayload(data);
      setSettingsNotice("");
    } catch (loadError) {
      setError(loadError.message || "Unable to load settings.");
    } finally {
      setSettingsLoading(false);
    }
  }

  function updateSettingsDraft(key, value) {
    setSettingsDraft((current) => ({ ...(current || {}), [key]: value }));
  }

  function toggleSettingsSymbol(targetSymbol) {
    setSettingsDraft((current) => {
      const next = current || {};
      const selected = Array.isArray(next.execution_symbols) ? next.execution_symbols : [];
      const exists = selected.includes(targetSymbol);
      return {
        ...next,
        execution_symbols: exists ? selected.filter((item) => item !== targetSymbol) : [...selected, targetSymbol],
      };
    });
  }

  async function saveRuntimeSettings() {
    if (!settingsDraft) {
      return;
    }
    setSettingsSaving(true);
    try {
      const data = await apiFetch("/execution/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settingsDraftPayload(settingsDraft)),
      });
      applySettingsPayload(data);
      setSettingsNotice("Settings saved.");
      await Promise.all([refreshSnapshot(), refreshPortfolio()]);
    } catch (saveError) {
      setError(saveError.message || "Unable to save settings.");
    } finally {
      setSettingsSaving(false);
    }
  }

  async function testSmtpSettings() {
    setSmtpTesting(true);
    try {
      const data = await apiFetch("/execution/settings/test-smtp", { method: "POST" });
      setSettingsNotice(`SMTP test sent to ${data.recipient_count || 0} recipient(s).`);
      await loadRuntimeSettings(false);
    } catch (testError) {
      setError(testError.message || "Unable to send SMTP test.");
    } finally {
      setSmtpTesting(false);
    }
  }

  async function refreshChart() {
    const data = await fetchChartPayload(symbol, chartRange, chartInterval);
    chartCacheRef.current.set(chartCacheKey(symbol, chartRange, chartInterval), data);
    setChart(data);
  }

  async function warmChartRange(range, interval = chartInterval) {
    if (!symbol || !range || !interval) {
      return;
    }
    const key = chartCacheKey(symbol, range, interval);
    if (chartCacheRef.current.has(key) || chartWarmRef.current.has(key)) {
      return;
    }
    chartWarmRef.current.add(key);
    try {
      const data = await fetchChartPayload(symbol, range, interval);
      chartCacheRef.current.set(key, data);
    } catch (_error) {
      // Hover/focus warming is opportunistic.
    } finally {
      chartWarmRef.current.delete(key);
    }
  }

  async function loadMoreChartHistory(before) {
    if (!symbol || !before || chartHistoryLoading) {
      return;
    }
    const requestKey = chartCacheKey(symbol, chartRange, chartInterval) + `::${before}`;
    if (chartHistoryWarmRef.current.has(requestKey)) {
      return;
    }
    chartHistoryWarmRef.current.add(requestKey);
    setChartHistoryLoading(true);
    try {
      const data = await apiChartFetch(
        `/api/candles/history?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(chartInterval)}&before=${encodeURIComponent(before)}&limit=${HISTORY_BATCH_SIZE}`,
      );
      const incoming = Array.isArray(data?.candles) ? data.candles : [];
      if (!incoming.length) {
        return;
      }
      const key = chartCacheKey(symbol, chartRange, chartInterval);
      setChart((current) => {
        const base = current || chartCacheRef.current.get(key) || selectedChart || {};
        const mergedCandles = mergeChartCandleRows(base.candles || [], incoming, "prepend");
        const nextChart = {
          ...base,
          candles: mergedCandles,
          available_count: Number(data?.available_count || base.available_count || 0) || null,
          oldest: mergedCandles[0]?.x || base.oldest,
          history_loaded_at: new Date().toISOString(),
        };
        chartCacheRef.current.set(key, nextChart);
        return nextChart;
      });
    } catch (loadError) {
      setError(loadError.message || "Unable to load historical candles.");
    } finally {
      chartHistoryWarmRef.current.delete(requestKey);
      setChartHistoryLoading(false);
    }
  }

  async function updateMode(mode) {
    setBusy(true);
    try {
      const response = await apiFetch("/execution/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      setRuntimeMode(response.mode || mode);
      setBrokerHealth(response.broker || null);
      startTransition(() => {
        setSnapshot((current) => ({
          ...(current || {}),
          execution: {
            ...(current?.execution || {}),
            mode: response.mode || mode,
            broker: response.broker || current?.execution?.broker,
          },
        }));
      });
      if (response.mode === "live" && response.broker?.status !== "ok") {
        const brokerError = parseBrokerError(response.broker?.errors?.[0]?.body)
          || parseBrokerError(response.broker?.errors?.[0])
          || "Live broker is not ready.";
        setError(`Live mode selected, but broker is not ready: ${brokerError}`);
      }
      await Promise.all([refreshSnapshot(), refreshPortfolio()]);
    } catch (actionError) {
      setError(actionError.message || "Unable to switch trading mode.");
    } finally {
      setBusy(false);
    }
  }

  async function resetPaperCapital() {
    const nextBalance = window.prompt("Reset paper capital to amount", String(snapshot?.stats?.paper_starting_balance || 500000));
    if (!nextBalance) {
      return;
    }
    setBusy(true);
    try {
      await apiFetch("/execution/paper/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ starting_balance: Number(nextBalance), clear_open_positions: true }),
      });
      await Promise.all([refreshSnapshot(), refreshPortfolio(), refreshTradeHistory()]);
    } catch (actionError) {
      setError(actionError.message || "Unable to reset paper capital.");
    } finally {
      setBusy(false);
    }
  }

  async function closePosition(positionId) {
    setBusy(true);
    try {
      await apiFetch(`/execution/positions/${positionId}/close`, { method: "POST" });
      await Promise.all([refreshSnapshot(), refreshChart()]);
    } catch (actionError) {
      setError(actionError.message || "Unable to close position.");
    } finally {
      setBusy(false);
    }
  }

  async function deletePosition(positionId) {
    setBusy(true);
    try {
      await apiFetch(`/execution/positions/${positionId}`, { method: "DELETE" });
      await Promise.all([refreshSnapshot(), refreshPortfolio(), refreshTradeHistory()]);
    } catch (actionError) {
      setError(actionError.message || "Unable to delete position.");
    } finally {
      setBusy(false);
    }
  }

  async function emergencyExit() {
    setBusy(true);
    try {
      await apiFetch("/execution/emergency-exit", { method: "POST" });
      await Promise.all([refreshSnapshot(), refreshChart()]);
    } catch (actionError) {
      setError(actionError.message || "Unable to exit positions.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    if (!symbol) {
      return;
    }
    refreshPortfolio();
  }, [symbol, runtimeMode]);

  useEffect(() => {
    if (activeView === "history") {
      refreshTradeHistory();
    }
  }, [activeView]);

  useEffect(() => {
    if (activeView === "settings") {
      loadRuntimeSettings(!settingsData);
    }
  }, [activeView]);

  useEffect(() => {
    if (!symbol || !["operations", "calendar", "database"].includes(activeView)) {
      return undefined;
    }
    let active = true;
    async function loadStaticSnapshot() {
      try {
        const data = await apiFetch(
          `/api/live/snapshot?symbol=${encodeURIComponent(symbol)}&include_static=true&include_chart=false&include_option=false`,
        );
        if (!active) {
          return;
        }
        startTransition(() => setSnapshot((current) => ({
          ...(current || {}),
          calendar: data.calendar || current?.calendar || {},
          history: data.history || current?.history || {},
          notifications: data.notifications || current?.notifications || {},
        })));
      } catch (loadError) {
        if (active) {
          setError(loadError.message || "Unable to load static market data.");
        }
      }
    }
    loadStaticSnapshot();
    return () => {
      active = false;
    };
  }, [activeView, symbol]);

  useEffect(() => {
    if (!replay.active || !replay.playing) {
      return undefined;
    }
    const delay = Math.max(16, 1000 / Math.max(1, Number(replay.speed || 1)));
    const timer = window.setInterval(() => {
      setReplay((current) => {
        const total = (chart?.candles || []).length;
        if (!current.active || current.index >= total - 1) {
          return { ...current, playing: false };
        }
        return { ...current, index: current.index + 1 };
      });
    }, delay);
    return () => window.clearInterval(timer);
  }, [replay.active, replay.playing, replay.speed, chart?.candles?.length]);

  useEffect(() => {
    const last = Number(snapshot?.price?.last);
    if (!Number.isFinite(last)) {
      return;
    }
    const triggered = (layout.alerts || []).filter((alert) => (
      alert.enabled
      && alert.symbol === symbol
      && alert.type === "price"
      && (alert.operator === ">" ? last > Number(alert.value) : last < Number(alert.value))
    ));
    if (!triggered.length) {
      return;
    }
    setAlertEvents((current) => [
      ...triggered.map((alert) => ({ ...alert, firedAt: new Date().toISOString(), last })),
      ...current,
    ].slice(0, 8));
    setLayout((current) => ({
      ...current,
      alerts: (current.alerts || []).map((alert) => triggered.some((item) => item.id === alert.id) ? { ...alert, enabled: false, fired_at: new Date().toISOString() } : alert),
    }));
    triggered.forEach((alert) => {
      fetch("/api/alerts/event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...alert, fired_at: new Date().toISOString(), last }),
      }).catch(() => {});
    });
  }, [snapshot?.price?.last, layout.alerts, symbol]);

  function updateCurrentIndicators(nextIndicators) {
    setLayout((current) => upsertSymbolState(current, symbol, {
      indicators: typeof nextIndicators === "function" ? nextIndicators(getSymbolState(current, symbol).indicators || []) : nextIndicators,
    }));
  }

  function updateCurrentDrawings(nextDrawings) {
    setLayout((current) => upsertSymbolState(current, symbol, {
      drawings: typeof nextDrawings === "function" ? nextDrawings(getSymbolState(current, symbol).drawings || []) : nextDrawings,
    }));
  }

  function selectSymbol(nextSymbol) {
    if (!nextSymbol) {
      return;
    }
    setSnapshot((current) => current ? {
      ...current,
      price: {},
      freshness: {
        ...(current.freshness || {}),
        market_status: "loading",
      },
    } : current);
    setChart(null);
    setPortfolio(null);
    setStreamState("connecting");
    setChartLoading(true);
    setError("");
    setLayout((current) => ({
      ...current,
      selectedSymbol: nextSymbol,
      recentSearches: addUnique(current.recentSearches, nextSymbol),
    }));
    setSearchOpen(false);
    setSearchQuery("");
    setReplay({ active: false, playing: false, speed: 1, index: 0 });
    setSelectedDrawingId("");
    setSymbol(nextSymbol);
  }

  function toggleFavorite(targetSymbol = symbol) {
    setLayout((current) => {
      const exists = current.favorites.includes(targetSymbol);
      return {
        ...current,
        favorites: exists ? current.favorites.filter((item) => item !== targetSymbol) : addUnique(current.favorites, targetSymbol, 100),
      };
    });
  }

  function addToWatchlist(targetSymbol = symbol) {
    setLayout((current) => {
      const active = current.activeWatchlist || "Indices";
      return {
        ...current,
        watchlists: {
          ...current.watchlists,
          [active]: addUnique(current.watchlists?.[active] || [], targetSymbol, 100),
        },
      };
    });
  }

  function removeFromWatchlist(targetSymbol) {
    setLayout((current) => {
      const active = current.activeWatchlist || "Indices";
      return {
        ...current,
        watchlists: {
          ...current.watchlists,
          [active]: (current.watchlists?.[active] || []).filter((item) => item !== targetSymbol),
        },
      };
    });
  }

  function setActiveWatchlist(name) {
    setLayout((current) => ({ ...current, activeWatchlist: name }));
  }

  function createWatchlist() {
    const name = window.prompt("Watchlist name", "New List");
    if (!name) {
      return;
    }
    setLayout((current) => ({
      ...current,
      activeWatchlist: name,
      watchlists: { ...current.watchlists, [name]: current.watchlists?.[name] || [] },
    }));
  }

  function createAlert(alert) {
    setLayout((current) => ({ ...current, alerts: [alert, ...(current.alerts || [])] }));
  }

  function saveTemplate(name) {
    setLayout((current) => ({
      ...current,
      templates: {
        ...current.templates,
        [name]: {
          indicators: getSymbolState(current, symbol).indicators || [],
          chartSettings: getSymbolState(current, symbol).chartSettings || {},
        },
      },
    }));
  }

  function applyTemplate(name) {
    const template = layout.templates?.[name];
    if (!template) {
      return;
    }
    setLayout((current) => upsertSymbolState(current, symbol, {
      indicators: template.indicators || [],
      chartSettings: template.chartSettings || {},
    }));
  }

  if (loading && !bootstrapped && !snapshot) {
    return <div className="loader">Loading trading workspace...</div>;
  }

  function updateHistoryFilter(key, value) {
    const next = { ...historyFilters, [key]: value };
    setHistoryFilters(next);
    refreshTradeHistory(next);
  }

  function changeChartRange(nextRange) {
    const cached = chartCacheRef.current.get(chartCacheKey(symbol, "all", "1minute"));
    if (cached) {
      setChart(cached);
    }
    setChartRange("all");
    setChartInterval("1minute");
  }

  function changeChartInterval(nextInterval) {
    setChartRange("all");
    setChartInterval("1minute");
  }

  return (
    <div className="workspace">
      <Sidebar
        activeView={activeView}
        onChange={setActiveView}
        snapshot={selectedSnapshot}
        streamState={streamState}
      />

      <main className="content-shell">
        <header className="terminal-header">
          <div className="terminal-topbar">
            <div className="topbar-title">
              <span className="material-symbols-outlined" aria-hidden="true">{activeMeta.icon}</span>
              <div>
                <p>{activeMeta.eyebrow}</p>
                <h2>{activeMeta.label}</h2>
              </div>
            </div>

            <div className="topbar-market">
              <label className="field compact-field">
                <span>Symbol</span>
                <select className="select" value={symbol} onChange={(event) => selectSymbol(event.target.value)}>
                  {symbols.map((item) => (
                    <option key={item} value={item}>{item}</option>
                  ))}
                </select>
              </label>
              <button type="button" className="icon-button" onClick={() => setSearchOpen(true)} title="Symbol search">
                <span className="material-symbols-outlined" aria-hidden="true">search</span>
              </button>
              <button type="button" className={`icon-button ${layout.favorites.includes(symbol) ? "active" : ""}`} onClick={() => toggleFavorite(symbol)} title="Favorite">
                <span className="material-symbols-outlined" aria-hidden="true">star</span>
              </button>
              <div className="market-stat">
                <span>Spot</span>
                <strong className={priceTone}>{formatMoney(price.last)}</strong>
              </div>
              <div className="market-stat">
                <span>Move</span>
                <strong className={priceTone}>{formatSignedMoney(price.change)}</strong>
              </div>
            </div>

            <div className="topbar-actions">
              <div className="mode-toggle">
                <button
                  type="button"
                  className={`toggle-button ${execution.mode === "paper" ? "active" : ""}`}
                  disabled={busy}
                  onClick={() => updateMode("paper")}
                >
                  Paper
                </button>
                <button
                  type="button"
                  className={`toggle-button ${execution.mode === "live" ? "active" : ""}`}
                  disabled={busy}
                  onClick={() => updateMode("live")}
                >
                  Live
                </button>
              </div>
              <button type="button" className="icon-button" disabled={busy} onClick={refreshSnapshot} title="Refresh snapshot">
                <span className="material-symbols-outlined" aria-hidden="true">refresh</span>
              </button>
              <button type="button" className="icon-button" disabled={busy || chartLoading} onClick={refreshChart} title="Refresh chart">
                <span className="material-symbols-outlined" aria-hidden="true">candlestick_chart</span>
              </button>
              <button
                type="button"
                className="icon-button theme-toggle"
                onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
                title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
                aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              >
                <span className="material-symbols-outlined" aria-hidden="true">
                  {theme === "dark" ? "light_mode" : "dark_mode"}
                </span>
              </button>
              <button type="button" className="danger-button compact-danger" disabled={busy} onClick={emergencyExit}>
                Emergency Exit
              </button>
            </div>
          </div>

          <div className="header-status-row">
            <span className="chip emphasis">Stream {streamState}</span>
            <span className={`chip ${brokerStatus === "ok" || brokerStatus === "paper" ? "emphasis" : ""}`}>
              Broker {brokerStatus}
            </span>
            <span className="chip">Market {freshness.market_status || "-"}</span>
            <span className="chip">Watchlist {layout.activeWatchlist}</span>
            {chartHistoryLoading ? <span className="chip">Loading history</span> : null}
            <span className="chip">IST {summaryLine}</span>
          </div>
          <div className="watchlist-row">
            {Object.keys(layout.watchlists || {}).map((name) => (
              <button key={name} type="button" className={`range-button ${layout.activeWatchlist === name ? "active" : ""}`} onClick={() => setActiveWatchlist(name)}>
                {name}
              </button>
            ))}
            <button type="button" className="tool-button" onClick={createWatchlist} title="New watchlist">
              <span className="material-symbols-outlined" aria-hidden="true">add</span>
            </button>
            <button type="button" className="line-button" onClick={() => addToWatchlist(symbol)}>Add {symbol}</button>
            {(layout.watchlists?.[layout.activeWatchlist] || []).map((item) => (
              <button key={item} type="button" className={`watchlist-chip ${item === symbol ? "active" : ""}`} onClick={() => selectSymbol(item)} onDoubleClick={() => removeFromWatchlist(item)}>
                {layout.favorites.includes(item) ? "★ " : ""}{item}
              </button>
            ))}
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}
        {searchOpen ? (
          <div className="modal-overlay symbol-search-overlay" onMouseDown={() => setSearchOpen(false)}>
            <div className="symbol-search-modal" onMouseDown={(event) => event.stopPropagation()}>
              <div className="panel-head">
                <div>
                  <h2>Symbol Search</h2>
                </div>
                <button type="button" className="line-button" onClick={() => setSearchOpen(false)}>Close</button>
              </div>
              <input
                className="symbol-search-input"
                autoFocus
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && searchResults[0]) {
                    selectSymbol(searchResults[0]);
                  }
                  if (event.key === "Escape") {
                    setSearchOpen(false);
                  }
                }}
                placeholder="Search symbol, company, or index"
              />
              <div className="search-section">
                <span>Results</span>
                {searchResults.map((item) => (
                  <button key={item} type="button" className="search-result" onClick={() => selectSymbol(item)}>
                    <strong>{item}</strong>
                    <small>{layout.favorites.includes(item) ? "Favorite" : "Symbol"}</small>
                  </button>
                ))}
              </div>
              <div className="search-columns">
                <div className="search-section">
                  <span>Recent</span>
                  {(layout.recentSearches || []).map((item) => (
                    <button key={item} type="button" className="search-pill" onClick={() => selectSymbol(item)}>{item}</button>
                  ))}
                </div>
                <div className="search-section">
                  <span>Favorites</span>
                  {(layout.favorites || []).map((item) => (
                    <button key={item} type="button" className="search-pill" onClick={() => selectSymbol(item)}>{item}</button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : null}
        {alertEvents.length ? (
          <div className="alert-stack">
            {alertEvents.map((event) => (
              <div key={`${event.id}-${event.firedAt}`} className="alert-toast">
                {event.symbol} price {event.operator} {formatMoney(event.value)} hit at {formatMoney(event.last)}
              </div>
            ))}
          </div>
        ) : null}

        {activeView === "overview" ? (
          <>
            <section className="metrics-grid">
              <MetricCard
                label="Current Move"
                value={`${formatSignedMoney(price.change)} / ${formatPct(price.change_pct)}`}
                meta={`Open ${formatMoney(price.open)} - High ${formatMoney(price.high)} - Low ${formatMoney(price.low)}`}
                tone={priceTone}
              />
              <MetricCard
                label="Today P&L"
                value={formatSignedMoney(stats.total_pnl_today)}
                meta={`${stats.wins_today || 0} wins / ${stats.total_trades_today || 0} trades`}
                tone={pnlTone}
              />
              <MetricCard
                label="Open Positions"
                value={String(stats.open_positions_count || 0)}
                meta={`${formatSignedMoney(stats.open_positions_unrealized_pnl || 0)} unrealized`}
                tone={Number(stats.open_positions_unrealized_pnl) >= 0 ? "positive" : "negative"}
              />
              <MetricCard
                label="Win Rate"
                value={formatPct(stats.win_rate)}
                meta={`Snapshot ${formatDateTime(selectedSnapshot.generated_at)}`}
                tone="neutral"
              />
            </section>

            <section className="chart-row">
              <ChartPanel
                symbol={selectedSnapshot.symbol || symbol}
                chart={selectedChart}
                rangeKey={chartRange}
                intervalKey={chartInterval}
                indicators={currentSymbolState.indicators || []}
                drawings={currentSymbolState.drawings || []}
                activeDrawingTool={activeDrawingTool}
                selectedDrawingId={selectedDrawingId}
                replay={replay}
                alerts={layout.alerts || []}
                onRangeChange={changeChartRange}
                onIntervalChange={changeChartInterval}
                onRangeWarm={warmChartRange}
                onLoadMoreHistory={loadMoreChartHistory}
                onIndicatorsChange={updateCurrentIndicators}
                onDrawingsChange={updateCurrentDrawings}
                onActiveDrawingToolChange={setActiveDrawingTool}
                onSelectedDrawingIdChange={setSelectedDrawingId}
                onReplayChange={setReplay}
                onCreateAlert={createAlert}
                onSaveTemplate={saveTemplate}
                onApplyTemplate={applyTemplate}
                theme={theme}
              />
            </section>

            <section className="signal-board">
              <StrategySignalCard signal={signal} option={option} />
              <EngineStrikeCard signal={signal} option={option} />
              <LiveDataCard
                price={price}
                freshness={freshness}
                history={history}
                chart={selectedChart}
                stream={stream}
              />
            </section>

            <section>
              <PositionsTable positions={deferredPositions} onClose={closePosition} onDelete={deletePosition} onInspect={setContractModal} />
            </section>

            <section>
              <TradesTable rows={selectedSnapshot.recent_trades || []} />
            </section>
          </>
        ) : null}

        {activeView === "operations" ? (
          <section className="support-grid operations-screen">
            <PortfolioCard
              mode={execution.mode || "paper"}
              portfolio={portfolio}
              onResetPaper={resetPaperCapital}
              onRefresh={refreshPortfolio}
              busy={busy || portfolioLoading}
            />
            <SessionCard calendar={calendar} option={option} freshness={freshness} />
            <DataCoverageCard history={history} />
            <RiskControlsCard execution={execution} stats={stats} />
          </section>
        ) : null}

        {activeView === "positions" ? (
          <section>
            <PositionTracker symbol={symbol} mode={execution.mode || "paper"} />
          </section>
        ) : null}

        {activeView === "optionchain" ? (
          <section>
            <OptionChain symbol={symbol} />
          </section>
        ) : null}

        {activeView === "history" ? (
          <TradeHistoryDashboard
            historyData={tradeHistory}
            strategyRows={strategyPerformance}
            filters={historyFilters}
            onFilterChange={updateHistoryFilter}
            loading={historyLoading}
          />
        ) : null}

        {activeView === "calendar" ? (
          <section className="double-grid">
            <TradingCalendar calendar={calendar} />
            <SessionCard calendar={calendar} option={option} freshness={freshness} />
          </section>
        ) : null}

        {activeView === "database" ? <HistoryWindow history={history} /> : null}

        {activeView === "settings" ? (
          <SettingsWindow
            data={settingsData}
            draft={settingsDraft}
            loading={settingsLoading}
            saving={settingsSaving}
            testingSmtp={smtpTesting}
            notice={settingsNotice}
            onDraftChange={updateSettingsDraft}
            onToggleSymbol={toggleSettingsSymbol}
            onSave={saveRuntimeSettings}
            onReload={() => loadRuntimeSettings(true)}
            onTestSmtp={testSmtpSettings}
          />
        ) : null}
      </main>
      <ContractChartModal contract={contractModal} onClose={() => setContractModal(null)} />
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
