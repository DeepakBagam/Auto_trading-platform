# Backtest Analysis Report - Nifty 50 Final Tuning

**Report Date:** April 25, 2026  
**Timezone:** Asia/Kolkata  
**Instrument:** Nifty 50  
**Timeframe:** 1-minute intraday

## Final Tuning Applied

The execution layer was tuned only for profit capture. Entry quality and risk caps were left structurally unchanged.

- Initial stop kept at `42` points
- Break-even trigger moved from `+25` to `+35`
- Trailing stop now activates from `+50` with a `20` point gap
- Partial exit reduced from `50%` to `30%` at `+50`
- Tightened stop updates now apply from the next 1-minute bar instead of re-stopping the trade inside the same bar
- Daily trade cap was expanded from `1` to `2`, but only with a strict second-trade cutoff at `11:00 IST`

## 30-Day Result

Source: [backtest_Nifty_50_20260425_005711_summary.json](/D:/AUTOMATED%20AI%20TRADING%20PLATFORM/logs/backtests/backtest_Nifty_50_20260425_005711_summary.json)

- Total trades: `26`
- Win rate: `61.5%`
- Profit factor: `2.54`
- Total PnL: `+416.01` points
- Avg trade: `+16.00` points
- Avg win: `+42.89` points
- Avg loss: `-27.00` points
- Max drawdown: `129.8` points
- Partial exit rate: `53.8%`
- Buy / Sell split: `13 BUY`, `13 SELL`
- Trade-day profile: `14` single-trade days, `6` two-trade days

## Before vs After

Baseline before final tuning was the prior stable one-trade run with:

- Win rate: `50.0%`
- Profit factor: `2.21`
- Avg trade: `+12.66`
- Avg win: `+46.33`
- Avg loss: `-21.00`
- Max drawdown: `130.5`
- Breakeven exits: `5`
- Daily cap: `1 trade/day`

After final tuning plus controlled second-trade enablement:

- Win rate improved to `61.5%`
- Profit factor improved to `2.54`
- Total PnL improved from `+337.02` points to `+416.01` points
- Avg trade remained strong at `+16.00`
- Breakeven exits remained limited at `3`
- Drawdown stayed effectively flat at `129.8`

Net effect: the second trade is helpful only when treated as an early-session extension trade. Raw unrestricted `2/day` degraded sharply and was rejected.

## Validation Check

A wider 60-day sanity run was also checked to reduce overfitting risk.

- Trades: `57`
- Win rate: `54.4%`
- Profit factor: `1.86`
- Total PnL: `+612.64`
- Max drawdown: `173.32`

Interpretation:

- The controlled second-trade rule improved both the 30-day and 60-day windows versus the one-trade version.
- The benefit comes from capturing early continuation moves, not from simply trading more.
- The 60-day result is still below elite quality, so this should be treated as a stronger paper-trading configuration, not a fully finished production profile.

## Recommended Default

Keep the current simplified working default:

- `SL = 42`
- `BE = +35`
- `Partial = 30% at +50`
- `Trail = activate at +50, gap 20`
- `Up to 2 trades per day`
- `Second trade only if triggered by 11:00 IST`
- `No overlapping positions`

This is the best low-complexity two-trade configuration tested in the current session.
