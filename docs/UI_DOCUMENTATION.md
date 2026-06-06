# UI Documentation - Realtime Options Trading Desk

## 📋 Table of Contents
1. [Overview](#overview)
2. [Technology Stack](#technology-stack)
3. [Screen Details](#screen-details)
4. [Components Reference](#components-reference)
5. [API Integration](#api-integration)
6. [Real-time Features](#real-time-features)
7. [File Structure](#file-structure)

---

## Overview

The trading platform UI is a **React-based single-page application** with 6 main screens for monitoring, analyzing, and executing options trades. It provides real-time market data via WebSocket and supports both paper and live trading modes.

**Total Screens:** 6  
**Components:** 25+ React components  
**Update Frequency:** 150ms (price) / 2s (positions)  
**Chart Library:** Lightweight Charts 4.2.0  

---

## Technology Stack

### Frontend
- **React 18** (Production, loaded via CDN)
- **ReactDOM 18** (Production, loaded via CDN)
- **Babel Standalone** (JSX transpilation in browser)
- **Lightweight Charts 4.2.0** (TradingView charting library)

### Styling
- **Custom CSS** (3 files: styles.css, OptionChain.css, PositionTracker.css)
- **Fonts:** Space Grotesk, IBM Plex Mono (Google Fonts)
- **Color Scheme:** Light theme (#f7f5ee background)

### Backend Integration
- **REST API** (FastAPI)
- **WebSocket** (Real-time price/position updates)
- **Server-Sent Events** (Alternative streaming)

---

## Screen Details

### 1. Live Desk (Overview Screen)
**Route ID:** `overview`  
**Purpose:** Main dashboard with comprehensive market view and trading controls

#### Layout Structure
```
Hero Panel (Symbol selector, Price display, Mode toggle)
↓
Status Row (Stream status, Market status, IST time)
↓
Metrics Grid (6 cards)
├─ Current Move
├─ Today P&L
├─ Open Positions
├─ Win Rate
├─ Paper Balance
└─ Daily Loss Limit
↓
Overview Grid (2 columns)
├─ Chart Panel (Left, full width)
│  ├─ Candlestick chart
│  ├─ Range selector (1D, 5D, 1M, 6M, 1Y, 2Y)
│  └─ Chart markers (entry/exit signals)
└─ Right Stack
   ├─ Strategy Signal Card
   ├─ Live Data Card
   ├─ Portfolio Card
   ├─ Session Card
   └─ Data Coverage Card
↓
Positions Table (Open positions with actions)
↓
Double Grid
├─ Trades Table (Recent closed trades)
└─ Orders Feed (Latest order events)
```

#### Key Features
- **Real-time price updates** via WebSocket
- **Interactive chart** with time range selection
- **Strategy signal checklist** (6 conditions):
  1. EMA 9 > EMA 21 cross
  2. RSI filter
  3. Signal candle direction
  4. Above-average volume
  5. Previous candle breakout
  6. Entry window open
- **Position management** (Exit/Delete actions)
- **Emergency exit** button for all positions
- **Mode toggle** (Paper ↔ Live)

#### API Endpoints Used
- `GET /api/live/snapshot?symbol={symbol}` - Initial snapshot
- `WS /api/live/ws?symbol={symbol}` - Real-time updates
- `GET /api/live/chart?symbol={symbol}&range={range}` - Chart data
- `POST /execution/emergency-exit` - Close all positions
- `GET /execution/portfolio` - Portfolio data

---

### 2. Positions Screen
**Route ID:** `positions`  
**Purpose:** Dedicated position tracking with real-time P&L monitoring

#### Layout Structure
```
Position Summary (7 cards)
├─ Today's P&L
├─ Unrealized P&L
├─ Open Positions
├─ Available Balance
├─ Invested Amount
├─ Paper Equity
└─ Win Rate
↓
Positions Table
├─ Contract (CE/PE badge, strike, expiry)
├─ Quantity
├─ Entry Price
├─ Current Price
├─ P&L (absolute + percentage)
├─ Stop Loss (editable inline)
├─ Target (editable inline)
├─ Entry Time
└─ Actions (Edit, Detail buttons)
```

#### Key Features
- **Live P&L updates** every 2 seconds
- **Inline SL/Target editing** with validation
- **Color-coded P&L** (green/red)
- **Position detail modal** with:
  - Entry/current status
  - Premium statistics
  - Premium history chart (SVG)
- **CE/PE badges** with color coding

#### Editable Fields
- **Stop Loss** - Can only tighten (increase), not loosen
- **Target Premium** - Must be above entry price

#### API Endpoints Used
- `GET /api/live/snapshot?symbol={symbol}` - Position data (every 2s)
- `POST /api/execution/update-sl-target` - Update SL/Target

---

### 3. Option Chain Screen
**Route ID:** `optionchain`  
**Purpose:** Professional option chain viewer (Upstox-style layout)

#### Layout Structure
```
Header
├─ Symbol + Spot Price + ATM Strike
└─ Expiry Selector Dropdown
↓
Option Chain Table (CE/PE Side-by-Side)
├─ CALL Side (CE)
│  ├─ OI (Open Interest)
│  ├─ Volume
│  ├─ IV (Implied Volatility)
│  ├─ LTP (Last Traded Price)
│  ├─ Change (absolute)
│  └─ Change% (percentage)
├─ STRIKE Column (Center, with ATM badge)
└─ PUT Side (PE)
   ├─ Change%
   ├─ Change
   ├─ LTP
   ├─ IV
   ├─ Volume
   └─ OI
↓
Footer
├─ Last Updated timestamp
└─ Total Strikes count
```

#### Key Features
- **Real-time updates** (1 second refresh)
- **ATM strike highlighting** with badge
- **ITM/OTM color coding**:
  - Strikes < ATM: ITM for CALL, OTM for PUT
  - Strikes > ATM: OTM for CALL, ITM for PUT
- **Click-to-expand** strike details modal
- **Expiry selector** (6 nearest expiries)
- **±10 strikes** around ATM displayed

#### Strike Detail Modal
Shows for both CE and PE:
- LTP
- Bid/Ask prices + Spread
- Greeks (Delta, Gamma, Theta, Vega)
- IV (Implied Volatility)
- OI (Open Interest)
- Volume

#### API Endpoints Used
- `GET /api/live/option-chain?symbol={symbol}&expiry={date}` - Chain data (every 1s)

---

### 4. Trade History Screen
**Route ID:** `history`  
**Purpose:** Historical trade analysis with filtering and performance metrics

#### Layout Structure
```
Filter Section
├─ Date From (date picker)
├─ Date To (date picker)
└─ Strategy Name (text input)
↓
Summary Cards (4 metrics)
├─ Total Trades
├─ Wins
├─ Losses
└─ Total P&L
↓
Double Grid
├─ Trade History Table
│  ├─ Date
│  ├─ Strategy
│  ├─ Contract (strike + type)
│  ├─ Entry Price
│  ├─ Exit Price
│  └─ P&L
└─ Strategy Performance Card
   ├─ Strategy Name
   ├─ Total Trades
   ├─ Win Rate
   ├─ Realized P&L
   └─ Max Drawdown
```

#### Key Features
- **Date range filtering**
- **Strategy filtering**
- **Auto-refresh on filter change**
- **P&L color coding** (green/red)
- **Strategy comparison** with win rate and drawdown
- **Last 300 trades** displayed

#### API Endpoints Used
- `GET /execution/trade-history?date_from={from}&date_to={to}&strategy={name}` - Trade data
- `GET /execution/strategy-performance` - Strategy stats

---

### 5. Trading Calendar Screen
**Route ID:** `calendar`  
**Purpose:** Monthly calendar view with trading days and option expiries

#### Layout Structure
```
Calendar Legend
├─ Trading Day (green dot)
├─ Closed Day (red dot)
├─ Expiry Day (yellow dot)
└─ Today (blue dot)
↓
Calendar Weekdays Header
[Mon] [Tue] [Wed] [Thu] [Fri] [Sat] [Sun]
↓
Calendar Grid (7 × ~5 rows)
├─ Day Number
├─ Weekday Name
└─ Status Indicators
↓
Session Card (Trading hours + upcoming days)
```

#### Key Features
- **Visual calendar grid** (Monday-first layout)
- **Multi-status indicators** (day can be both trading + expiry)
- **Today highlighting**
- **Next 6 upcoming trading days** listed
- **Market session hours** (IST timezone)
- **Holiday/weekend marking**

#### Calendar States
- **Trading Day** - Regular market hours
- **Closed** - Weekend/holiday
- **Expiry** - Option expiry date
- **Today** - Current day highlight

---

### 6. Database Window Screen
**Route ID:** `database`  
**Purpose:** Data retention, coverage, and database statistics

#### Layout Structure
```
Metrics Grid (4 cards)
├─ Retention Window (2 years)
├─ Option Quotes Count
├─ Orders Count
└─ Closed Trades Count
↓
Market Data Coverage Table
├─ Interval (1minute, 30minute, day)
├─ Rows (record count)
├─ Oldest Timestamp
├─ Latest Timestamp
└─ Coverage Status (ready/filling)
```

#### Key Features
- **Rolling 2-year retention** window
- **Data coverage by interval**:
  - 1-minute candles
  - 30-minute candles
  - Daily candles
- **Timestamp display in IST**
- **Coverage status** (ready vs filling)
- **Option quotes tracking**

---

## Components Reference

### Core App Components

#### `App` (Main Component)
**Props:** None  
**State:** 
- `symbol` - Current selected symbol
- `snapshot` - Live snapshot data
- `chart` - Chart data
- `portfolio` - Portfolio state
- `activeView` - Current screen (overview/positions/etc.)
- `streamState` - WebSocket connection state

**Key Methods:**
- `refreshSnapshot()` - Reload snapshot from API
- `refreshPortfolio()` - Reload portfolio data
- `updateMode(mode)` - Switch paper/live mode
- `closePosition(id)` - Close single position
- `emergencyExit()` - Close all positions

---

#### `Sidebar`
**Props:** `activeView`, `onChange`, `snapshot`, `streamState`  
**Purpose:** Navigation sidebar with system status

**Displays:**
- Navigation buttons (6 screens)
- Stream status (live/reconnecting/down)
- Source status (live/stopped)
- Today's date (IST)
- Retention period
- Email notification status

---

#### `ChartPanel`
**Props:** `symbol`, `chart`, `loading`, `rangeKey`, `onRangeChange`  
**Purpose:** Candlestick chart with time range selector

**Features:**
- Lightweight Charts integration
- Auto-resize on window resize
- Range buttons (1D, 5D, 1M, 6M, 1Y, 2Y)
- Chart markers for trades
- Live scrolling for real-time data
- IST timezone formatting

**Chart Options:**
- Candlestick colors: Green (#0d8a62), Red (#c4563d)
- Background: #f7f5ee
- Auto-fit on range change
- Time scale with IST labels

---

#### `MetricCard`
**Props:** `label`, `value`, `meta`, `tone`  
**Purpose:** Reusable metric display card

**Tones:** `positive`, `negative`, `neutral`

---

#### `StrategySignalCard`
**Props:** `signal`, `option`  
**Purpose:** Display 6-condition strategy checklist

**Conditions Displayed:**
1. EMA 9 > EMA 21 cross (with current values)
2. RSI filter (with RSI value)
3. Signal candle direction (with O/C prices)
4. Above-average volume (with vol/avg)
5. Previous candle breakout (with prev high/low)
6. Entry window open (with IST time window)

**Visual States:** `on` (green), `off` (red), `neutral` (gray)

---

#### `PositionTracker`
**Props:** `symbol`  
**Purpose:** Live position tracking component

**Features:**
- Auto-refresh every 2 seconds
- Inline SL/Target editing
- Position detail modal with premium chart
- P&L color coding
- CE/PE badges

---

#### `OptionChain`
**Props:** `symbol`  
**Purpose:** Option chain viewer component

**Features:**
- Auto-refresh every 1 second
- Expiry selector
- Strike detail modal
- CE/PE side-by-side layout
- ATM strike highlighting

---

### Utility Functions

#### Price Formatting
```javascript
formatMoney(value) // ₹1,234.56
formatSignedMoney(value) // +₹1,234.56 or -₹1,234.56
formatPct(value) // 12.34%
formatCount(value) // 1,234,567
```

#### Time Formatting
```javascript
formatDateTime(value) // 27/04/2026, 09:30:15 (IST)
formatDate(value) // 27 Apr 2026
formatTime(value) // 09:30
formatAge(seconds) // 2m 15s
```

#### Chart Helpers
```javascript
parseChartTime(value) // ISO → Unix timestamp
chartTimeToDate(timeValue) // Chart time → Date object
formatChartTickMark(timeValue) // Chart axis label
formatChartCrosshairTime(timeValue) // Crosshair tooltip
```

---

## API Integration

### REST Endpoints

#### Live Data
```
GET /api/live/symbols
GET /api/live/snapshot?symbol={symbol}
GET /api/live/chart?symbol={symbol}&range={range}
GET /api/live/option-chain?symbol={symbol}&expiry={date}
GET /api/live/option-contract-chart?symbol={symbol}&expiry={date}&strike={strike}&option_type={CE|PE}&position_id={id}
GET /api/live/dashboard-state
```

#### Execution
```
GET /execution/status
GET /execution/mode
POST /execution/mode (body: {mode: "paper"|"live"})
GET /execution/portfolio
POST /execution/emergency-exit
POST /execution/positions/{id}/close
DELETE /execution/positions/{id}
POST /execution/paper/reset (body: {starting_balance: 500000, clear_open_positions: true})
POST /execution/update-sl-target (body: {position_id, new_sl, new_target})
```

#### Analytics
```
GET /execution/trade-history?date_from={date}&date_to={date}&strategy={name}
GET /execution/strategy-performance
GET /execution/audit-logs?limit={count}
```

---

### WebSocket Protocol

#### Connection
```
WS ws://localhost:8000/api/live/ws?symbol={symbol}
```

#### Message Types

**1. Snapshot (Full State)**
```json
{
  "type": "snapshot",
  "payload": {
    "generated_at": "2026-01-15T09:30:15+05:30",
    "symbol": "Nifty 50",
    "price": {"last": 23456.78, "change": +45.67, "change_pct": 0.19},
    "signal": {...},
    "positions": [...],
    "stats": {...},
    "calendar": {...},
    "history": {...}
  }
}
```

**2. Price Update (Quick Tick)**
```json
{
  "type": "price",
  "payload": {
    "generated_at": "2026-01-15T09:30:16+05:30",
    "symbol": "Nifty 50",
    "price": {"last": 23457.12, "change": +46.01},
    "candle": {
      "x": "2026-01-15T09:30:00+05:30",
      "open": 23450.00,
      "high": 23460.00,
      "low": 23445.00,
      "close": 23457.12
    }
  }
}
```

**3. Error**
```json
{
  "type": "error",
  "payload": {
    "detail": "Error message"
  }
}
```

#### Update Frequencies
- **Price ticks:** 150ms (configurable via `ui_tick_interval_ms`)
- **Full snapshot:** 800ms (configurable via `ui_stream_interval_ms`)
- **Position refresh:** 2000ms (component-level)
- **Option chain:** 1000ms (component-level)

---

## Real-time Features

### WebSocket Connection Management

#### Auto-Reconnect Strategy
```
Initial connect → Open
       ↓
   Connection lost
       ↓
State: "reconnecting"
       ↓
Exponential backoff: 1s, 2s, 4s, 8s (max 10s)
       ↓
Retry connection → Back to "live"
```

#### Connection States
- **connecting** - Initial connection attempt
- **live** - Connected and receiving data
- **reconnecting** - Lost connection, attempting to reconnect
- **down** - Multiple failures, long delay before retry

---

### Live Data Merging

#### Quick Price Update Merge
```javascript
function mergeQuickUpdate(currentSnapshot, update) {
  return {
    ...currentSnapshot,
    price: update?.price || currentSnapshot.price,
    freshness: update?.freshness || currentSnapshot.freshness,
    stream: update?.stream || currentSnapshot.stream
  };
}
```

#### Live Chart Update Merge
```javascript
function mergeLiveChart(currentChart, update) {
  const nextCandles = [...currentChart.candles];
  const nextCandle = update?.candle;
  
  if (nextCandle?.x) {
    // Update last candle if same timestamp, else append
    const lastIndex = nextCandles.length - 1;
    if (nextCandles[lastIndex]?.x === nextCandle.x) {
      nextCandles[lastIndex] = nextCandle;
    } else {
      nextCandles.push(nextCandle);
    }
  }
  
  // Apply window limit
  const windowSize = LIVE_CHART_WINDOW_LIMITS[currentChart.range] || 500;
  return {
    ...currentChart,
    candles: nextCandles.slice(-windowSize)
  };
}
```

---

### Chart Window Limits
```javascript
const LIVE_CHART_WINDOW_LIMITS = {
  "1d": 500,   // 500 minutes = ~8 hours
  "5d": 2500,  // 2500 minutes = ~41 hours
};
```

---

## File Structure

```
web/
├── index.html              # HTML shell, loads React + dependencies
├── app.js                  # Main app (3,200+ lines)
│   ├── App component
│   ├── Sidebar
│   ├── ChartPanel
│   ├── MetricCard
│   ├── StrategySignalCard
│   ├── LiveDataCard
│   ├── SessionCard
│   ├── TradingCalendar
│   ├── HistoryWindow
│   ├── PositionsTable
│   ├── TradesTable
│   ├── OrdersFeed
│   ├── PortfolioCard
│   ├── StrategyPerformanceCard
│   ├── TradeHistoryDashboard
│   ├── ContractChartModal
│   └── Utility functions (30+)
├── OptionChain.js          # Option chain component (~400 lines)
│   ├── OptionChain component
│   └── StrikeDetailModal component
├── PositionTracker.js      # Position tracker (~550 lines)
│   ├── PositionTracker component
│   └── PositionDetailModal component
├── styles.css              # Base styles
├── OptionChain.css         # Option chain styles
└── PositionTracker.css     # Position tracker styles
```

---

## Screen Navigation Map

```
Sidebar Navigation
├── [Overview] Live Desk
│   ├── Chart + Signals
│   ├── Positions
│   ├── Trades
│   └── Orders
├── [Trading] Positions
│   ├── Summary Cards
│   ├── Positions Table
│   └── Detail Modal
├── [Options] Option Chain
│   ├── Chain Table
│   └── Strike Detail Modal
├── [Analytics] Trade History
│   ├── Filters
│   ├── Trade Table
│   └── Strategy Performance
├── [Calendar] Trading Calendar
│   ├── Calendar Grid
│   └── Session Info
└── [Database] Database Window
    ├── Retention Stats
    └── Coverage Table
```

---

## Color Coding Reference

### Price/P&L States
- **Positive (Profit):** Green (#0d8a62)
- **Negative (Loss):** Red (#c4563d)
- **Neutral:** Gray (#173042)

### Stream States
- **Live:** Green dot
- **Reconnecting:** Yellow/Orange dot
- **Down:** Red dot

### Option Chain
- **ATM Strike:** Bold with badge
- **ITM (Call) / OTM (Put):** Light background
- **OTM (Call) / ITM (Put):** Light background
- **CE (Call):** Green accent
- **PE (Put):** Red accent

### Position Badges
- **CE (Call):** Green badge (#00b16a)
- **PE (Put):** Red badge (#e63946)

---

## Key Metrics Displayed

### Live Desk Metrics
1. **Current Move** - Price change (absolute + percentage)
2. **Today P&L** - Total realized + unrealized P&L
3. **Open Positions** - Count + unrealized P&L
4. **Win Rate** - Percentage of winning trades
5. **Paper Balance** - Available + invested + equity
6. **Daily Loss Limit** - Maximum allowed loss

### Position Summary Metrics
1. **Today's P&L** - Realized P&L today
2. **Unrealized P&L** - Open position P&L
3. **Open Positions** - Count of open positions
4. **Available Balance** - Cash available
5. **Invested Amount** - Capital in use
6. **Paper Equity** - Total account value
7. **Win Rate** - Win percentage + W/L count

### Trade History Metrics
1. **Total Trades** - Count
2. **Wins** - Winning trades count
3. **Losses** - Losing trades count
4. **Total P&L** - Sum of all P&L

### Strategy Performance Metrics
1. **Strategy Name** - Strategy identifier
2. **Total Trades** - Count per strategy
3. **Win Rate** - Percentage per strategy
4. **Realized P&L** - Total profit/loss
5. **Max Drawdown** - Largest peak-to-trough decline

---

## Timezone Handling

All timestamps are displayed in **IST (Asia/Kolkata)** timezone.

```javascript
const IST_TIME_ZONE = "Asia/Kolkata";

// Usage
date.toLocaleString("en-IN", { 
  hour12: false, 
  timeZone: IST_TIME_ZONE 
});
```

---

## Modal Components

### 1. Contract Chart Modal
**Trigger:** Click on strike price in positions/orders  
**Shows:**
- LTP history chart (SVG line chart)
- P&L history chart (SVG line chart)
- Entry/current/quantity stats
- Time-series data (last 500 quotes)

### 2. Strike Detail Modal (Option Chain)
**Trigger:** Click on any strike row  
**Shows:**
- CE and PE side-by-side
- Bid/Ask/Spread
- Greeks (Delta, Gamma, Theta, Vega)
- IV, OI, Volume

### 3. Position Detail Modal
**Trigger:** Click "Detail" button in positions table  
**Shows:**
- Entry details (premium, time, quantity, strategy)
- Current status (premium, P&L, SL, target)
- Statistics (max/min premium, max profit/drawdown)
- Premium history chart (SVG)

---

## Performance Optimizations

### React Optimizations
- **useDeferredValue** for positions list (non-blocking updates)
- **startTransition** for large state updates (chart data, snapshots)
- **useRef** for chart instances (avoid re-renders)
- **useRef** for WebSocket cache (persistent across renders)

### Chart Optimizations
- **setData** vs **update** - Full replace vs incremental
- **fitContent** on range change only
- **scrollToRealTime** for live updates
- **Window limits** to cap memory usage

### WebSocket Optimizations
- **Digest comparison** - Only send updates when data changes
- **Tick vs Snapshot** - 150ms ticks, 800ms full snapshots
- **Auto-reconnect** with exponential backoff

---

## Browser Compatibility

**Minimum Requirements:**
- Modern browser with ES6+ support
- WebSocket support
- CSS Grid/Flexbox support

**Tested Browsers:**
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

---

## Summary

**Total Lines of Code:** ~4,500+ lines  
**Total Components:** 25+  
**Total Screens:** 6  
**Total Modals:** 3  
**API Endpoints Used:** 15+  
**WebSocket Streams:** 1  
**Update Frequency:** 150ms (fastest)  
**Chart Library:** Lightweight Charts 4.2.0  
**No Build Step:** Pure React via CDN  

This is a **production-ready, professional-grade trading UI** with real-time capabilities, comprehensive position management, and advanced analytics.
