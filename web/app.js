const { startTransition, useDeferredValue, useEffect, useRef, useState } = React;
const { createRoot } = ReactDOM;
const { LightweightCharts } = window;

const NAV_ITEMS = [
  { id: "overview", label: "Live Desk", eyebrow: "Overview" },
  { id: "calendar", label: "Trading Calendar", eyebrow: "Calendar" },
  { id: "database", label: "Database Window", eyebrow: "Database" },
];
const IST_TIME_ZONE = "Asia/Kolkata";
const CHART_RANGE_FALLBACK = [
  { key: "1d", label: "1D Live", interval: "1minute", supports_live: true },
  { key: "5d", label: "5D", interval: "1minute", supports_live: true },
  { key: "1m", label: "1M", interval: "30minute", supports_live: false },
  { key: "6m", label: "6M", interval: "day", supports_live: false },
  { key: "1y", label: "1Y", interval: "day", supports_live: false },
  { key: "2y", label: "2Y", interval: "day", supports_live: false },
];

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

function chartCacheKey(symbol, rangeKey) {
  return `${symbol || ""}::${rangeKey || "1d"}`;
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    cache: "no-store",
    headers: { "Cache-Control": "no-cache" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "Request failed");
  }
  return response.json();
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

  const windowSize = Math.max((current.candles || []).length, 1);
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

function Sidebar({ activeView, onChange, snapshot, streamState }) {
  const calendar = snapshot?.calendar || {};
  const history = snapshot?.history || {};
  const notifications = snapshot?.notifications || {};
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
        <p className="eyebrow">Realtime Desk</p>
        <h1>Trading UI</h1>
        <p>Simple, fast, and wired to the live stream in IST.</p>
      </div>

      <nav className="nav-card">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`nav-button ${activeView === item.id ? "active" : ""}`}
            onClick={() => onChange(item.id)}
          >
            <small>{item.eyebrow}</small>
            <strong>{item.label}</strong>
          </button>
        ))}
      </nav>

      <div className="sidebar-card">
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

function ChartPanel({ symbol, chart, loading, rangeKey, onRangeChange }) {
  const hostRef = useRef(null);
  const chartRef = useRef(null);
  const seriesRef = useRef(null);
  const deferredChart = useDeferredValue(chart);
  const ranges = deferredChart?.available_ranges || CHART_RANGE_FALLBACK;

  useEffect(() => {
    if (!hostRef.current || !LightweightCharts) {
      return undefined;
    }
    const instance = LightweightCharts.createChart(hostRef.current, {
      width: hostRef.current.clientWidth,
      height: hostRef.current.clientHeight,
      layout: { background: { color: "#f7f5ee" }, textColor: "#173042" },
      rightPriceScale: { borderColor: "rgba(18, 52, 69, 0.18)" },
      timeScale: { borderColor: "rgba(18, 52, 69, 0.18)", timeVisible: true, secondsVisible: false },
      grid: {
        vertLines: { color: "rgba(18, 52, 69, 0.08)" },
        horzLines: { color: "rgba(18, 52, 69, 0.08)" },
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
      upColor: "#0d8a62",
      downColor: "#c4563d",
      wickUpColor: "#0d8a62",
      wickDownColor: "#c4563d",
      borderVisible: false,
    });
    chartRef.current = instance;
    seriesRef.current = series;

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
      instance.remove();
      chartRef.current = null;
      seriesRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!seriesRef.current || !deferredChart) {
      return;
    }
    const candles = (deferredChart.candles || [])
      .map((row) => ({
        time: parseChartTime(row.x),
        open: Number(row.open),
        high: Number(row.high),
        low: Number(row.low),
        close: Number(row.close),
      }))
      .filter((row) => Number.isFinite(row.time));
    seriesRef.current.setData(candles);
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
  }, [deferredChart]);

  return (
    <article className="panel chart-panel">
      <div className="panel-head">
        <div>
          <h2>{symbol} chart</h2>
          <p>
            {deferredChart?.is_resampled
              ? "Stored one-minute history, displayed as larger candles for long windows."
              : "Streaming one-minute candles over WebSocket for the live window."}
          </p>
        </div>
        <div className="chart-meta">
          <span className="chip">
            {deferredChart?.label || "Chart"}
            {" / "}
            {deferredChart?.interval || "-"}
          </span>
          <span className={`chip ${deferredChart?.supports_live ? "emphasis" : ""}`}>
            {deferredChart?.supports_live ? "Live WS" : "Archive"}
          </span>
        </div>
      </div>
      <div className="range-row">
        {ranges.map((item) => (
          <button
            key={item.key}
            type="button"
            className={`range-button ${rangeKey === item.key ? "active" : ""}`}
            onClick={() => onRangeChange(item.key)}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="status-row chart-status-row">
        <span className="chip">From {formatDate(deferredChart?.start_date)}</span>
        <span className="chip">To {formatDate(deferredChart?.end_date)}</span>
        <span className="chip">{loading ? "Loading range" : `${formatCount(deferredChart?.candles?.length)} candles`}</span>
      </div>
      <div ref={hostRef} className="chart-canvas" />
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
          <p>Data-only workspace with WebSocket prices and IST candles.</p>
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

      <div className="stack-list">
        {!freshness?.is_live ? (
          <div className="note-row">
            {sourceRunning
              ? "The UI socket is up, but the source feed is stale. Check market session timing and the upstream stream freshness."
              : "The UI can only show stored candles because the market-stream runtime is not running in this process right now."}
          </div>
        ) : null}
        {runtime?.last_error ? (
          <div className="note-row">Last stream error: {runtime.last_error}</div>
        ) : null}
        <div className="note-row">Automated entries are disabled.</div>
        <div className="note-row">The screen now stays focused on live data, positions, orders, and trades.</div>
      </div>
    </article>
  );
}

function DataCoverageCard({ history }) {
  const intervals = (history?.intervals || []).slice(0, 3);
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Coverage</h2>
          <p>Latest retained candles in IST for the active database window.</p>
        </div>
      </div>

      <div className="stack-list">
        {intervals.map((row) => (
          <div key={row.interval} className="note-row compact">
            <strong>{row.interval}</strong>
            <span>{formatCount(row.count)} rows</span>
            <span>{formatDateTime(row.latest_ts)}</span>
            <span>{row.window_ready ? "ready" : "filling"}</span>
          </div>
        ))}
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
          <p>Exchange day, next session, and expiry watch in IST.</p>
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
  const intervals = history?.intervals || [];
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
            <h2>Market Data Coverage</h2>
            <p>The stream keeps a rolling two-year window in the database and shows all timestamps in IST.</p>
          </div>
        </div>

        <div className="table-shell">
          <table>
            <thead>
              <tr>
                <th>Interval</th>
                <th>Rows</th>
                <th>Oldest</th>
                <th>Latest</th>
                <th>Coverage</th>
              </tr>
            </thead>
            <tbody>
              {intervals.map((row) => (
                <tr key={row.interval}>
                  <td>{row.interval}</td>
                  <td>{formatCount(row.count)}</td>
                  <td>{formatDateTime(row.oldest_ts)}</td>
                  <td>{formatDateTime(row.latest_ts)}</td>
                  <td>
                    <span className={`tag ${row.window_ready ? "buy" : "hold"}`}>
                      {row.window_ready ? "ready" : "filling"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}

function PositionsTable({ positions, onClose }) {
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
                  <td>{row.strike}</td>
                  <td>{row.option_type}</td>
                  <td>{formatMoney(row.entry_premium)}</td>
                  <td>{formatMoney(row.current_premium)}</td>
                  <td className={Number(row.unrealized_pnl) >= 0 ? "positive" : "negative"}>
                    {formatSignedMoney(row.unrealized_pnl)}
                  </td>
                  <td>{formatMoney(row.current_sl)}</td>
                  <td>
                    <button type="button" className="line-button" onClick={() => onClose(row.position_id)}>
                      Exit
                    </button>
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

function OrdersFeed({ rows }) {
  return (
    <article className="panel">
      <div className="panel-head">
        <div>
          <h2>Recent Orders</h2>
          <p>Latest order events.</p>
        </div>
      </div>
      {!rows.length ? (
        <div className="empty-state">No orders yet.</div>
      ) : (
        <div className="stack-list">
          {rows.map((row) => (
            <div key={row.id} className="note-row">
              <div className="row-split">
                <strong>{row.order_kind} {row.option_type || ""}</strong>
                <small>{formatDateTime(row.created_at)}</small>
              </div>
              <div className="chip-row">
                <span className="chip">{row.side}</span>
                <span className="chip">{row.quantity} qty</span>
                <span className="chip">{row.status}</span>
              </div>
              <div>
                {row.strike_price ? `${row.symbol} ${row.strike_price} ${row.option_type || ""}` : row.symbol}
                {" - "}
                {formatMoney(row.price)}
                {" - "}
                {row.exit_reason || row.consensus_reason || "active"}
              </div>
            </div>
          ))}
        </div>
      )}
    </article>
  );
}

function App() {
  const [symbols, setSymbols] = useState([]);
  const [symbol, setSymbol] = useState("");
  const [snapshot, setSnapshot] = useState(null);
  const [chart, setChart] = useState(null);
  const [chartRange, setChartRange] = useState("1d");
  const [chartLoading, setChartLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [streamState, setStreamState] = useState("connecting");
  const [busy, setBusy] = useState(false);
  const [activeView, setActiveView] = useState("overview");
  const chartCacheRef = useRef(new Map());

  useEffect(() => {
    let cancelled = false;
    async function loadSymbols() {
      try {
        const payload = await apiFetch("/api/live/symbols");
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
    loadSymbols();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!symbol) {
      return undefined;
    }
    let active = true;
    let socket;

    async function loadInitial() {
      try {
        setLoading(true);
        const data = await apiFetch(`/api/live/snapshot?symbol=${encodeURIComponent(symbol)}`);
        if (!active) {
          return;
        }
        startTransition(() => {
          setSnapshot(data);
          if (data?.chart?.range) {
            chartCacheRef.current.set(chartCacheKey(symbol, data.chart.range), data.chart);
            if (chartRange === data.chart.range) {
              setChart(data.chart);
            }
          }
        });
        setError("");
      } catch (loadError) {
        if (active) {
          setError(loadError.message || "Unable to load snapshot.");
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
        }
      });
      socket.addEventListener("message", (event) => {
        if (!active) {
          return;
        }
        try {
          const message = JSON.parse(event.data);
          if (message.type === "snapshot") {
            startTransition(() => setSnapshot((current) => ({
              ...(current || {}),
              ...(message.payload || {}),
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
              setChart((current) => mergeLiveChart(current, message.payload));
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
        }
      });
      socket.addEventListener("error", () => {
        if (active) {
          setStreamState("reconnecting");
        }
      });
    }

    loadInitial();
    return () => {
      active = false;
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
        setChartLoading(true);
        const data = await apiFetch(
          `/api/live/chart?symbol=${encodeURIComponent(symbol)}&range=${encodeURIComponent(chartRange)}`,
        );
        if (!active) {
          return;
        }
        startTransition(() => {
          chartCacheRef.current.set(chartCacheKey(symbol, chartRange), data);
          setChart(data);
        });
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

    const cached = chartCacheRef.current.get(chartCacheKey(symbol, chartRange));
    if (cached) {
      startTransition(() => setChart(cached));
      if (!cached.supports_live) {
        return () => {
          active = false;
        };
      }
    }

    loadChart();
    return () => {
      active = false;
    };
  }, [symbol, chartRange]);

  const selectedSnapshot = snapshot || {};
  const selectedChart = chart || selectedSnapshot.chart || {};
  const stats = selectedSnapshot.stats || {};
  const price = selectedSnapshot.price || {};
  const freshness = selectedSnapshot.freshness || {};
  const option = selectedSnapshot.option || {};
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

  async function refreshSnapshot() {
    const data = await apiFetch(`/api/live/snapshot?symbol=${encodeURIComponent(symbol)}`);
    startTransition(() => setSnapshot(data));
  }

  async function refreshChart() {
    const data = await apiFetch(
      `/api/live/chart?symbol=${encodeURIComponent(symbol)}&range=${encodeURIComponent(chartRange)}`,
    );
    startTransition(() => {
      chartCacheRef.current.set(chartCacheKey(symbol, chartRange), data);
      setChart(data);
    });
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

  if (loading && !snapshot) {
    return <div className="loader">Loading trading workspace...</div>;
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
        <header className="hero-panel">
          <div>
            <p className="eyebrow">{activeMeta.eyebrow}</p>
            <h2>{selectedSnapshot.symbol || symbol || "Trading desk"}</h2>
            <p className="hero-copy">
              React interface with live WebSocket updates, trading-calendar visibility, and a rolling
              two-year database window in IST.
            </p>
          </div>

          <div className="hero-actions">
            <label className="field">
              <span>Symbol</span>
              <select className="select" value={symbol} onChange={(event) => setSymbol(event.target.value)}>
                {symbols.map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>

            <div className="price-box">
              <span>Live price</span>
              <strong className={priceTone}>{formatMoney(price.last)}</strong>
              <small>{formatSignedMoney(price.change)} / {formatPct(price.change_pct)}</small>
            </div>

            <div className="button-row">
              <button type="button" className="secondary-button" disabled={busy} onClick={refreshSnapshot}>
                Refresh
              </button>
              <button type="button" className="secondary-button" disabled={busy || chartLoading} onClick={refreshChart}>
                Refresh chart
              </button>
              <button type="button" className="danger-button" disabled={busy} onClick={emergencyExit}>
                Emergency exit
              </button>
            </div>
          </div>
        </header>

        <div className="status-row">
          <span className="chip emphasis">Stream {streamState}</span>
          <span className="chip">Market {freshness.market_status || "-"}</span>
          <span className="chip">IST {summaryLine}</span>
        </div>

        {error ? <div className="error-banner">{error}</div> : null}

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

            <section className="overview-grid">
              <ChartPanel
                symbol={selectedSnapshot.symbol || symbol}
                chart={selectedChart}
                loading={chartLoading}
                rangeKey={chartRange}
                onRangeChange={setChartRange}
              />
              <div className="right-stack">
                <LiveDataCard
                  price={price}
                  freshness={freshness}
                  history={history}
                  chart={selectedChart}
                  stream={stream}
                />
                <SessionCard calendar={calendar} option={option} freshness={freshness} />
                <DataCoverageCard history={history} />
              </div>
            </section>

            <section>
              <PositionsTable positions={deferredPositions} onClose={closePosition} />
            </section>

            <section className="double-grid">
              <TradesTable rows={selectedSnapshot.recent_trades || []} />
              <OrdersFeed rows={selectedSnapshot.recent_orders || []} />
            </section>
          </>
        ) : null}

        {activeView === "calendar" ? (
          <section className="double-grid">
            <TradingCalendar calendar={calendar} />
            <SessionCard calendar={calendar} option={option} freshness={freshness} />
          </section>
        ) : null}

        {activeView === "database" ? <HistoryWindow history={history} /> : null}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
