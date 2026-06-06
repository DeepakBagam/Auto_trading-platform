// PositionTracker.js - Live Position Tracking with Real-time P&L
// Professional execution monitoring with risk metrics

const PositionTracker = ({ symbol, mode = "paper" }) => {
  const [positions, setPositions] = React.useState([]);
  const [summary, setSummary] = React.useState(null);
  const [loading, setLoading] = React.useState(true);
  const [selectedPosition, setSelectedPosition] = React.useState(null);
  const [editingPosition, setEditingPosition] = React.useState(null);

  const parseBrokerError = (value) => {
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
  };

  // Fetch positions and summary
  const fetchPositions = async () => {
    try {
      setLoading(true);
      const [snapshotResponse, modeResponse, portfolioResponse] = await Promise.all([
        fetch(`/api/live/snapshot?symbol=${encodeURIComponent(symbol || 'Nifty 50')}&include_static=false&include_chart=false&include_option=false`),
        fetch('/execution/mode'),
        fetch('/execution/portfolio'),
      ]);
      const data = await snapshotResponse.json();
      const modePayload = modeResponse?.ok ? await modeResponse.json() : null;
      const portfolio = portfolioResponse?.ok ? await portfolioResponse.json() : null;
      const activeMode = modePayload?.mode || portfolio?.mode || mode || data.execution?.mode || 'paper';
      const liveBrokerError = parseBrokerError(portfolio?.errors?.[0]?.body)
        || parseBrokerError(portfolio?.errors?.[0])
        || parseBrokerError(modePayload?.broker?.errors?.[0]?.body)
        || parseBrokerError(modePayload?.broker?.errors?.[0]);
      const liveReady = activeMode === 'live' && portfolio?.status === 'ok' && !liveBrokerError;
      const funds = portfolio?.funds || {};
      const liveAvailable = Number(
        funds.available_margin
        ?? funds.available_funds
        ?? funds.available_cash
        ?? funds.cash
        ?? funds.margin_available
        ?? 0
      );
      const liveUsed = Number(
        funds.utilised_margin
        ?? funds.used_margin
        ?? funds.margin_used
        ?? 0
      );
      const paperPositions = data.positions || [];
      const livePositions = Array.isArray(portfolio?.positions) ? portfolio.positions : [];

      setPositions(activeMode === 'live' ? livePositions : paperPositions);
      setSummary({
        mode: activeMode,
        portfolio_status: portfolio?.status || null,
        broker_error: activeMode === 'live' ? liveBrokerError : null,
        total_pnl: data.stats?.total_pnl_today || 0,
        open_positions: activeMode === 'live' ? livePositions.length : data.stats?.open_positions_count || 0,
        unrealized_pnl: activeMode === 'live'
          ? livePositions.reduce((sum, row) => sum + Number(row.pnl || row.unrealised || row.unrealized_pnl || 0), 0)
          : data.stats?.open_positions_unrealized_pnl || 0,
        total_trades: data.stats?.total_trades_today || 0,
        wins: data.stats?.wins_today || 0,
        win_rate: data.stats?.win_rate || 0,
        available_balance: activeMode === 'live' ? (liveReady ? liveAvailable : null) : data.stats?.paper_available_balance || 0,
        invested_amount: activeMode === 'live' ? (liveReady ? liveUsed : null) : data.stats?.paper_invested_amount || 0,
        equity: activeMode === 'live' ? (liveReady ? liveAvailable + liveUsed : null) : data.stats?.paper_equity || 0,
      });
    } catch (error) {
      console.error('Failed to fetch positions:', error);
    } finally {
      setLoading(false);
    }
  };

  const normalizePosition = (position) => ({
    position_id: position.position_id || position.id || position.instrument_token || position.tradingsymbol || position.symbol,
    option_type: position.option_type || position.instrument_type || "-",
    strike: position.strike || position.strike_price || position.tradingsymbol || position.symbol || "-",
    expiry: position.expiry || position.expiry_date || position.expiry_date_time,
    quantity: position.quantity ?? position.qty ?? 0,
    entry_premium: position.entry_premium ?? position.average_price ?? position.avg_price ?? 0,
    current_premium: position.current_premium ?? position.last_price ?? position.ltp ?? position.average_price ?? 0,
    unrealized_pnl: position.unrealized_pnl ?? position.pnl ?? position.unrealised ?? 0,
    current_sl: position.current_sl ?? position.stop_loss ?? 0,
    target_premium: position.target_premium ?? position.take_profit ?? 0,
    entry_time: position.entry_time || position.opened_at,
    strategy_name: position.strategy_name || position.product || position.product_type,
    raw: position,
  });

  React.useEffect(() => {
    fetchPositions();
    const interval = setInterval(fetchPositions, mode === 'live' ? 15000 : 10000);
    return () => clearInterval(interval);
  }, [symbol, mode]);

  const formatCurrency = (num) => {
    if (num === null || num === undefined) return '—';
    const sign = num >= 0 ? '+' : '';
    return `${sign}₹${Math.abs(num).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPercent = (num) => {
    if (num === null || num === undefined) return '0.00%';
    const sign = num >= 0 ? '+' : '';
    return `${sign}${num.toFixed(2)}%`;
  };

  const getPnLClass = (pnl) => {
    if (pnl > 0) return 'pnl-positive';
    if (pnl < 0) return 'pnl-negative';
    return 'pnl-neutral';
  };

  const calculatePnLPercent = (position) => {
    const entry = position.entry_premium || 0;
    const current = position.current_premium || entry;
    if (entry === 0) return 0;
    return ((current - entry) / entry) * 100;
  };

  const handleEditSLTarget = (position) => {
    setEditingPosition({
      position_id: position.position_id,
      current_sl: position.current_sl,
      target_premium: position.target_premium,
      entry_premium: position.entry_premium,
      current_premium: position.current_premium,
    });
  };

  const handleSaveSLTarget = async () => {
    if (!editingPosition) return;
    
    try {
      const response = await fetch('/execution/update-sl-target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          position_id: editingPosition.position_id,
          new_sl: parseFloat(editingPosition.current_sl),
          new_target: parseFloat(editingPosition.target_premium),
        }),
      });
      
      if (response.ok) {
        setEditingPosition(null);
        fetchPositions();
      } else {
        const error = await response.json();
        alert(`Failed to update: ${error.detail || 'Unknown error'}`);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };

  return (
    <div className="position-tracker-container">
      {loading && !summary ? (
        <div className="no-positions">
          <p>Loading positions...</p>
        </div>
      ) : null}
      {/* Summary Cards */}
      {summary && (
        <div className="position-summary">
          <div className="summary-card">
            <div className="summary-label">Today's P&L</div>
            <div className={`summary-value ${getPnLClass(summary.total_pnl)}`}>
              {formatCurrency(summary.total_pnl)}
            </div>
          </div>
          
          <div className="summary-card">
            <div className="summary-label">Unrealized P&L</div>
            <div className={`summary-value ${getPnLClass(summary.unrealized_pnl)}`}>
              {formatCurrency(summary.unrealized_pnl)}
            </div>
          </div>
          
          <div className="summary-card">
            <div className="summary-label">Open Positions</div>
            <div className="summary-value">{summary.open_positions}</div>
          </div>

          <div className="summary-card">
            <div className="summary-label">{summary.mode === 'live' ? 'Live Funds' : 'Available Balance'}</div>
            <div className="summary-value">
              {formatCurrency(summary.available_balance)}
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-label">{summary.mode === 'live' ? 'Used Margin' : 'Invested Amount'}</div>
            <div className="summary-value">
              {formatCurrency(summary.invested_amount)}
            </div>
          </div>

          <div className="summary-card">
            <div className="summary-label">{summary.mode === 'live' ? 'Live Equity' : 'Paper Equity'}</div>
            <div className={`summary-value ${getPnLClass((summary.equity || 0) - (summary.available_balance || 0) - (summary.invested_amount || 0))}`}>
              {formatCurrency(summary.equity)}
            </div>
          </div>
          
          <div className="summary-card">
            <div className="summary-label">Win Rate</div>
            <div className="summary-value">
              {summary.win_rate.toFixed(1)}% ({summary.wins}/{summary.total_trades})
            </div>
          </div>
        </div>
      )}
      {summary?.mode === 'live' && summary?.broker_error ? (
        <div className="no-positions broker-warning">
          Broker not ready: {summary.broker_error}
        </div>
      ) : null}

      {/* Positions Table */}
      <div className="positions-table-wrapper">
        <div className="positions-header">
          <h3>Open Positions</h3>
          <span className={`live-indicator ${summary?.mode === 'live' ? '' : 'paper-mode'}`}>
            ● {String(summary?.mode || 'paper').toUpperCase()}
          </span>
        </div>

        {positions.length === 0 ? (
          <div className="no-positions">
            <p>No open positions</p>
          </div>
        ) : (
          <table className="positions-table">
            <thead>
              <tr>
                <th>Contract</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>Current</th>
                <th>P&L</th>
                <th>P&L %</th>
                <th>SL</th>
                <th>Target</th>
                <th>Time</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((sourcePos) => {
                const pos = normalizePosition(sourcePos);
                const pnlPercent = calculatePnLPercent(pos);
                const isEditing = editingPosition?.position_id === pos.position_id;
                
                return (
                  <tr key={pos.position_id} className="position-row">
                    <td className="contract-cell">
                      <div className="contract-info">
                        <span className={`option-badge ${pos.option_type === 'CE' ? 'call-badge' : 'put-badge'}`}>
                          {pos.option_type}
                        </span>
                        <span className="strike-label">{pos.strike}</span>
                      </div>
                      <div className="contract-meta">
                        {new Date(pos.expiry).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })}
                      </div>
                    </td>
                    
                    <td className="qty-cell">{pos.quantity}</td>
                    
                    <td className="entry-cell">
                      ₹{(pos.entry_premium || 0).toFixed(2)}
                    </td>
                    
                    <td className="current-cell">
                      <strong>₹{(pos.current_premium || 0).toFixed(2)}</strong>
                    </td>
                    
                    <td className={`pnl-cell ${getPnLClass(pos.unrealized_pnl)}`}>
                      <strong>{formatCurrency(pos.unrealized_pnl || 0)}</strong>
                    </td>
                    
                    <td className={`pnl-pct-cell ${getPnLClass(pnlPercent)}`}>
                      {formatPercent(pnlPercent)}
                    </td>
                    
                    <td className="sl-cell">
                      {isEditing ? (
                        <input
                          type="number"
                          step="0.05"
                          value={editingPosition.current_sl}
                          onChange={(e) => setEditingPosition({...editingPosition, current_sl: e.target.value})}
                          className="sl-input"
                        />
                      ) : (
                        <span className="sl-value">₹{(pos.current_sl || 0).toFixed(2)}</span>
                      )}
                    </td>
                    
                    <td className="target-cell">
                      {isEditing ? (
                        <input
                          type="number"
                          step="0.05"
                          value={editingPosition.target_premium}
                          onChange={(e) => setEditingPosition({...editingPosition, target_premium: e.target.value})}
                          className="target-input"
                        />
                      ) : (
                        <span className="target-value">₹{(pos.target_premium || 0).toFixed(2)}</span>
                      )}
                    </td>
                    
                    <td className="time-cell">
                      {pos.entry_time ? new Date(pos.entry_time).toLocaleTimeString('en-IN', { 
                        hour: '2-digit', 
                        minute: '2-digit' 
                      }) : '-'}
                    </td>
                    
                    <td className="actions-cell">
                      {isEditing ? (
                        <div className="action-buttons">
                          <button className="btn-save" onClick={handleSaveSLTarget}>✓</button>
                          <button className="btn-cancel" onClick={() => setEditingPosition(null)}>✕</button>
                        </div>
                      ) : (
                        <div className="action-buttons">
                          <button className="btn-edit" onClick={() => handleEditSLTarget(pos)}>Edit</button>
                          <button className="btn-detail" onClick={() => setSelectedPosition(pos)}>Detail</button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Position Detail Modal */}
      {selectedPosition && (
        <PositionDetailModal 
          position={selectedPosition}
          onClose={() => setSelectedPosition(null)}
        />
      )}
    </div>
  );
};

// Position Detail Modal with Premium History Chart
const PositionDetailModal = ({ position, onClose }) => {
  const premiumHistory = position.premium_history || [];
  
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-IN', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const calculateStats = () => {
    if (premiumHistory.length === 0) return null;
    
    const premiums = premiumHistory.map(h => h.premium);
    const maxPremium = Math.max(...premiums);
    const minPremium = Math.min(...premiums);
    const currentPremium = position.current_premium || 0;
    const entryPremium = position.entry_premium || 0;
    
    return {
      max: maxPremium,
      min: minPremium,
      range: maxPremium - minPremium,
      maxDrawdown: entryPremium - minPremium,
      maxProfit: maxPremium - entryPremium,
    };
  };

  const stats = calculateStats();

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="position-detail-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Position Details - {position.option_type} {position.strike}</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="modal-body">
          <div className="position-detail-grid">
            {/* Entry Details */}
            <div className="detail-section">
              <h4>Entry Details</h4>
              <div className="detail-row">
                <span>Entry Premium:</span>
                <strong>₹{(position.entry_premium || 0).toFixed(2)}</strong>
              </div>
              <div className="detail-row">
                <span>Entry Time:</span>
                <span>{position.entry_time ? new Date(position.entry_time).toLocaleString('en-IN') : '-'}</span>
              </div>
              <div className="detail-row">
                <span>Quantity:</span>
                <span>{position.quantity}</span>
              </div>
              <div className="detail-row">
                <span>Strategy:</span>
                <span>{position.strategy_name || 'N/A'}</span>
              </div>
            </div>

            {/* Current Status */}
            <div className="detail-section">
              <h4>Current Status</h4>
              <div className="detail-row">
                <span>Current Premium:</span>
                <strong>₹{(position.current_premium || 0).toFixed(2)}</strong>
              </div>
              <div className="detail-row">
                <span>Unrealized P&L:</span>
                <strong className={position.unrealized_pnl >= 0 ? 'text-green' : 'text-red'}>
                  ₹{(position.unrealized_pnl || 0).toFixed(2)}
                </strong>
              </div>
              <div className="detail-row">
                <span>Stop Loss:</span>
                <span>₹{(position.current_sl || 0).toFixed(2)}</span>
              </div>
              <div className="detail-row">
                <span>Target:</span>
                <span>₹{(position.target_premium || 0).toFixed(2)}</span>
              </div>
            </div>

            {/* Statistics */}
            {stats && (
              <div className="detail-section">
                <h4>Statistics</h4>
                <div className="detail-row">
                  <span>Max Premium:</span>
                  <span>₹{stats.max.toFixed(2)}</span>
                </div>
                <div className="detail-row">
                  <span>Min Premium:</span>
                  <span>₹{stats.min.toFixed(2)}</span>
                </div>
                <div className="detail-row">
                  <span>Max Profit:</span>
                  <span className="text-green">₹{stats.maxProfit.toFixed(2)}</span>
                </div>
                <div className="detail-row">
                  <span>Max Drawdown:</span>
                  <span className="text-red">₹{stats.maxDrawdown.toFixed(2)}</span>
                </div>
              </div>
            )}
          </div>

          {/* Premium History Chart */}
          {premiumHistory.length > 0 && (
            <div className="premium-history-section">
              <h4>Premium History</h4>
              <div className="premium-chart">
                <svg width="100%" height="200" viewBox="0 0 600 200">
                  {/* Draw premium line */}
                  {premiumHistory.map((point, idx) => {
                    if (idx === 0) return null;
                    const prev = premiumHistory[idx - 1];
                    const x1 = (idx - 1) * (600 / (premiumHistory.length - 1));
                    const x2 = idx * (600 / (premiumHistory.length - 1));
                    const yScale = 180 / (stats.max - stats.min || 1);
                    const y1 = 190 - ((prev.premium - stats.min) * yScale);
                    const y2 = 190 - ((point.premium - stats.min) * yScale);
                    
                    return (
                      <line
                        key={idx}
                        x1={x1}
                        y1={y1}
                        x2={x2}
                        y2={y2}
                        stroke="#00b16a"
                        strokeWidth="2"
                      />
                    );
                  })}
                  
                  {/* Draw entry line */}
                  <line
                    x1="0"
                    y1={190 - ((position.entry_premium - stats.min) * (180 / (stats.max - stats.min || 1)))}
                    x2="600"
                    y2={190 - ((position.entry_premium - stats.min) * (180 / (stats.max - stats.min || 1)))}
                    stroke="#888"
                    strokeWidth="1"
                    strokeDasharray="5,5"
                  />
                </svg>
              </div>
              <div className="chart-legend">
                <span className="legend-item">
                  <span className="legend-color" style={{backgroundColor: '#00b16a'}}></span>
                  Premium
                </span>
                <span className="legend-item">
                  <span className="legend-color" style={{backgroundColor: '#888', opacity: 0.5}}></span>
                  Entry Level
                </span>
              </div>
            </div>
          )}
        </div>
        
        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};
