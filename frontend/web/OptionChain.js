// OptionChain.js - Professional Option Chain Component
// Upstox-style layout with CE/PE side-by-side

const OptionChain = ({ symbol, onInspectContract }) => {
  const AUTO_REFRESH_MS = 1000;
  const [chainData, setChainData] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState("");
  const [selectedStrike, setSelectedStrike] = React.useState(null);
  const [expiryDate, setExpiryDate] = React.useState(null);
  const inFlightRef = React.useRef(false);
  const mountedRef = React.useRef(true);

  // Fetch option chain data
  const fetchChain = async (showLoader = false, forceRefresh = false) => {
    if (inFlightRef.current || (!showLoader && document.visibilityState !== 'visible')) {
      return;
    }
    try {
      inFlightRef.current = true;
      if (showLoader) {
        setLoading(true);
      }
      const params = new URLSearchParams({
        symbol: symbol || 'Nifty 50',
        refresh: forceRefresh ? 'true' : 'false',
        strikes_each_side: '3',
      });
      if (expiryDate) params.append('expiry', expiryDate);
      
      const response = await fetch(`/api/live/option-chain?${params}`, {
        cache: 'no-store',
        headers: { Accept: 'application/json' },
      });
      if (!response.ok) {
        throw new Error(`Option chain request failed: ${response.status}`);
      }
      const data = await response.json();
      if (mountedRef.current) {
        setChainData(data);
        setError("");
      }
    } catch (error) {
      console.error('Failed to fetch option chain:', error);
      if (mountedRef.current) {
        setError(error.message || 'Failed to load option chain');
      }
    } finally {
      inFlightRef.current = false;
      if (mountedRef.current) {
        setLoading(false);
      }
    }
  };

  React.useEffect(() => {
    mountedRef.current = true;
    fetchChain(true, false);
    const refreshVisibleChain = () => fetchChain(false, false);
    const interval = window.setInterval(refreshVisibleChain, AUTO_REFRESH_MS);
    document.addEventListener('visibilitychange', refreshVisibleChain);
    return () => {
      mountedRef.current = false;
      window.clearInterval(interval);
      document.removeEventListener('visibilitychange', refreshVisibleChain);
    };
  }, [symbol, expiryDate]);

  if (loading && !chainData) {
    return (
      <div className="option-chain-loading">
        <div className="spinner"></div>
        <p>Loading option chain...</p>
      </div>
    );
  }

  if (!chainData) {
    return <div className="option-chain-error">{error || 'Failed to load option chain'}</div>;
  }

  // Group chain data by strike
  const strikeMap = new Map();
  chainData.chain.forEach(row => {
    const strike = row.strike;
    if (!strikeMap.has(strike)) {
      strikeMap.set(strike, { strike, ce: null, pe: null });
    }
    const entry = strikeMap.get(strike);
    if (row.ce) entry.ce = row.ce;
    if (row.pe) entry.pe = row.pe;
  });

  const strikes = Array.from(strikeMap.values()).sort((a, b) => a.strike - b.strike);

  const formatNumber = (num) => {
    if (num === null || num === undefined) return '-';
    return num.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const formatLargeNumber = (num) => {
    if (num === null || num === undefined) return '-';
    if (num >= 100000) return `${(num / 100000).toFixed(1)}L`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const formatIV = (num) => {
    if (num === null || num === undefined) return '-';
    const value = Number(num);
    if (!Number.isFinite(value)) return '-';
    return `${value.toFixed(1)}%`;
  };

  const getStrikeClass = (strike) => {
    const diff = Math.abs(strike - chainData.atm_strike);
    if (diff === 0) return 'atm-strike';
    if (strike < chainData.atm_strike) return 'itm-call otm-put';
    return 'otm-call itm-put';
  };

  const handleStrikeClick = (strike) => {
    setSelectedStrike(strike);
  };

  const openContractChart = (event, strike, optionType, quote) => {
    event.stopPropagation();
    if (!onInspectContract || !quote?.ltp) {
      return;
    }
    onInspectContract({
      symbol: chainData.symbol || symbol || 'Nifty 50',
      expiry: chainData.expiry_date,
      strike,
      optionType,
      entryPrice: quote.ltp,
    });
  };

  return (
    <div className="option-chain-container">
      {/* Header */}
      <div className="option-chain-header">
        <div className="chain-info">
          <h2>{chainData.symbol} Option Chain</h2>
          <div className="chain-meta">
            <span className="spot-price">
              Spot: <strong>{formatNumber(chainData.spot_price)}</strong>
            </span>
            <span className="atm-strike">
              ATM: <strong>{chainData.atm_strike}</strong>
            </span>
            <span className="chain-source">
              Source: {chainData.chain_source}
            </span>
          </div>
        </div>
        
        <div className="expiry-selector">
          <button type="button" className="chain-refresh-button" onClick={() => fetchChain(true, true)}>
            Refresh
          </button>
          <label>Expiry:</label>
          <select 
            value={expiryDate || chainData.expiry_date} 
            onChange={(e) => setExpiryDate(e.target.value)}
          >
            {chainData.available_expiries.map(exp => (
              <option key={exp} value={exp}>
                {new Date(exp).toLocaleDateString('en-IN', { 
                  day: '2-digit', 
                  month: 'short', 
                  year: 'numeric',
                  timeZone: 'Asia/Kolkata',
                })}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Option Chain Table */}
      <div className="option-chain-table-wrapper">
        <table className="option-chain-table">
          <thead>
            <tr>
              <th colSpan="6" className="call-header">CALL (CE)</th>
              <th className="strike-header">STRIKE</th>
              <th colSpan="6" className="put-header">PUT (PE)</th>
            </tr>
            <tr>
              <th>OI</th>
              <th>Vol</th>
              <th>IV</th>
              <th>LTP</th>
              <th>Chg</th>
              <th>Chg%</th>
              <th className="strike-header">Price</th>
              <th>Chg%</th>
              <th>Chg</th>
              <th>LTP</th>
              <th>IV</th>
              <th>Vol</th>
              <th>OI</th>
            </tr>
          </thead>
          <tbody>
            {strikes.map(({ strike, ce, pe }) => {
              const isATM = strike === chainData.atm_strike;
              const rowClass = `${getStrikeClass(strike)} ${isATM ? 'atm-row' : ''}`;
              
              return (
                <tr 
                  key={strike} 
                  className={rowClass}
                  onClick={() => handleStrikeClick(strike)}
                >
                  {/* CALL side */}
                  <td className="oi-cell">{formatLargeNumber(ce?.oi)}</td>
                  <td className="vol-cell">{formatLargeNumber(ce?.volume)}</td>
                  <td className="iv-cell">{formatIV(ce?.iv)}</td>
                  <td className="ltp-cell call-ltp">
                    <button
                      type="button"
                      className="strike-chart-button call"
                      disabled={!ce?.ltp}
                      onClick={(event) => openContractChart(event, strike, 'CE', ce)}
                      title={`Open ${strike} CE chart`}
                    >
                      {formatNumber(ce?.ltp)}
                    </button>
                  </td>
                  <td className="chg-cell">{ce?.ltp && ce?.close_price ? formatNumber(ce.ltp - ce.close_price) : '-'}</td>
                  <td className="chg-pct-cell">
                    {ce?.ltp && ce?.close_price ? 
                      `${((ce.ltp - ce.close_price) / ce.close_price * 100).toFixed(2)}%` : '-'}
                  </td>
                  
                  {/* STRIKE */}
                  <td className="strike-cell">
                    <strong>{strike}</strong>
                    {isATM && <span className="atm-badge">ATM</span>}
                  </td>
                  
                  {/* PUT side */}
                  <td className="chg-pct-cell">
                    {pe?.ltp && pe?.close_price ? 
                      `${((pe.ltp - pe.close_price) / pe.close_price * 100).toFixed(2)}%` : '-'}
                  </td>
                  <td className="chg-cell">{pe?.ltp && pe?.close_price ? formatNumber(pe.ltp - pe.close_price) : '-'}</td>
                  <td className="ltp-cell put-ltp">
                    <button
                      type="button"
                      className="strike-chart-button put"
                      disabled={!pe?.ltp}
                      onClick={(event) => openContractChart(event, strike, 'PE', pe)}
                      title={`Open ${strike} PE chart`}
                    >
                      {formatNumber(pe?.ltp)}
                    </button>
                  </td>
                  <td className="iv-cell">{formatIV(pe?.iv)}</td>
                  <td className="vol-cell">{formatLargeNumber(pe?.volume)}</td>
                  <td className="oi-cell">{formatLargeNumber(pe?.oi)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Strike Detail Modal */}
      {selectedStrike && (
        <StrikeDetailModal 
          strike={selectedStrike}
          chainData={chainData}
          onInspectContract={onInspectContract}
          onClose={() => setSelectedStrike(null)}
        />
      )}

      {/* Last Updated */}
      <div className="chain-footer">
        <span className="last-updated">
          Last updated: {chainData.chain_generated_at ? 
            new Date(chainData.chain_generated_at).toLocaleTimeString('en-IN', {
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
              hour12: false,
              timeZone: 'Asia/Kolkata',
            }) :
            'N/A'}
        </span>
        <span className="total-strikes">
          Showing {chainData.total_strikes} strikes
        </span>
      </div>
    </div>
  );
};

// Strike Detail Modal Component
const StrikeDetailModal = ({ strike, chainData, onInspectContract, onClose }) => {
  const strikeData = chainData.chain.find(row => row.strike === strike);
  
  if (!strikeData) return null;

  const ce = strikeData.ce || {};
  const pe = strikeData.pe || {};
  const openChart = (optionType, quote) => {
    if (!onInspectContract || !quote?.ltp) {
      return;
    }
    onClose();
    onInspectContract({
      symbol: chainData.symbol || 'Nifty 50',
      expiry: chainData.expiry_date,
      strike,
      optionType,
      entryPrice: quote.ltp,
    });
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="strike-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{strike} Strike Details</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="modal-body">
          <div className="strike-chart-actions">
            <button
              type="button"
              className="chain-refresh-button"
              disabled={!ce?.ltp}
              onClick={() => openChart('CE', ce)}
            >
              CE Chart
            </button>
            <button
              type="button"
              className="chain-refresh-button"
              disabled={!pe?.ltp}
              onClick={() => openChart('PE', pe)}
            >
              PE Chart
            </button>
          </div>
          <div className="strike-detail-grid">
            {/* CALL Details */}
            <div className="option-detail call-detail">
              <h4>CALL (CE)</h4>
              <div className="detail-row">
                <span>LTP:</span>
                <strong>{ce.ltp || '-'}</strong>
              </div>
              <div className="detail-row">
                <span>Bid/Ask:</span>
                <span>{ce.bid || '-'} / {ce.ask || '-'}</span>
              </div>
              <div className="detail-row">
                <span>Spread:</span>
                <span>{ce.bid && ce.ask ? (ce.ask - ce.bid).toFixed(2) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>IV:</span>
                <span>{ce.iv ? `${(ce.iv * 100).toFixed(2)}%` : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Delta:</span>
                <span>{ce.delta ? ce.delta.toFixed(4) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Gamma:</span>
                <span>{ce.gamma ? ce.gamma.toFixed(4) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Theta:</span>
                <span>{ce.theta ? ce.theta.toFixed(4) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Vega:</span>
                <span>{ce.vega ? ce.vega.toFixed(4) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>OI:</span>
                <span>{ce.oi ? ce.oi.toLocaleString('en-IN') : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Volume:</span>
                <span>{ce.volume ? ce.volume.toLocaleString('en-IN') : '-'}</span>
              </div>
            </div>

            {/* PUT Details */}
            <div className="option-detail put-detail">
              <h4>PUT (PE)</h4>
              <div className="detail-row">
                <span>LTP:</span>
                <strong>{pe.ltp || '-'}</strong>
              </div>
              <div className="detail-row">
                <span>Bid/Ask:</span>
                <span>{pe.bid || '-'} / {pe.ask || '-'}</span>
              </div>
              <div className="detail-row">
                <span>Spread:</span>
                <span>{pe.bid && pe.ask ? (pe.ask - pe.bid).toFixed(2) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>IV:</span>
                <span>{pe.iv ? `${(pe.iv * 100).toFixed(2)}%` : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Delta:</span>
                <span>{pe.delta ? pe.delta.toFixed(4) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Gamma:</span>
                <span>{pe.gamma ? pe.gamma.toFixed(4) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Theta:</span>
                <span>{pe.theta ? pe.theta.toFixed(4) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Vega:</span>
                <span>{pe.vega ? pe.vega.toFixed(4) : '-'}</span>
              </div>
              <div className="detail-row">
                <span>OI:</span>
                <span>{pe.oi ? pe.oi.toLocaleString('en-IN') : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Volume:</span>
                <span>{pe.volume ? pe.volume.toLocaleString('en-IN') : '-'}</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};
