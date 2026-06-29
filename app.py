"""
Streamlit visualizer — single-symbol strategy comparison + portfolio mode.

Run with:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetchers import YFinanceFetcher
from backtesting.engine import BacktestEngine
from backtesting.portfolio_engine import PortfolioBacktestEngine, PortfolioBacktestResult
from backtesting.results import BacktestResult
from strategies.buy_and_hold import BuyAndHold
from strategies.sma_crossover import SMACrossover
from strategies.rsi import RSIMeanReversion
from strategies.equal_weight import EqualWeightPortfolio
from strategies.multi_sma import MultiSMAPortfolio

st.set_page_config(page_title="AI Trading Gym", layout="wide", page_icon="📈")

# ── Strategy registries ───────────────────────────────────────────────────────

SINGLE_STRATEGIES = {
    "Buy & Hold":            lambda: BuyAndHold(),
    "SMA Crossover (10/30)": lambda: SMACrossover(fast=10, slow=30),
    "SMA Crossover (20/50)": lambda: SMACrossover(fast=20, slow=50),
    "RSI Mean Reversion":    lambda: RSIMeanReversion(period=14, oversold=30, overbought=70),
    "RSI (fast, tight)":     lambda: RSIMeanReversion(period=7,  oversold=25, overbought=75),
}

PORTFOLIO_STRATEGIES = {
    "Equal Weight (monthly rebal)": lambda: EqualWeightPortfolio(rebalance_days=21),
    "Equal Weight (quarterly)":     lambda: EqualWeightPortfolio(rebalance_days=63),
    "Multi-SMA Crossover (10/30)":  lambda: MultiSMAPortfolio(fast=10, slow=30),
    "Multi-SMA Crossover (20/50)":  lambda: MultiSMAPortfolio(fast=20, slow=50),
}

SORT_COL = {
    "Sharpe Ratio":   "Sharpe",
    "Total Return":   "Total Return %",
    "Ann. Return":    "Ann. Return %",
    "Sortino Ratio":  "Sortino",
    "Calmar Ratio":   "Calmar",
}


# ── Data fetching (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching market data...")
def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    return YFinanceFetcher().fetch(symbol, start=start, end=end, interval="1d", cache=False)


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _equity_fig(curves: dict[str, pd.Series], title: str) -> go.Figure:
    fig = go.Figure()
    for name, curve in curves.items():
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve.values, name=name, mode="lines",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Portfolio Value ($)",
                      hovermode="x unified", height=420,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def _drawdown_fig(curves: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for name, curve in curves.items():
        dd = (curve - curve.cummax()) / curve.cummax() * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name, mode="lines", fill="tozeroy",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.1f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown (%)",
                      hovermode="x unified", height=300,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def _price_trades_fig(data: pd.DataFrame, symbol: str, result: BacktestResult) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["open"], high=data["high"],
        low=data["low"], close=data["close"], name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350", showlegend=False,
    ), row=1, col=1)

    buys  = [f for f in result.fills if f.order.qty > 0 and f.timestamp is not None]
    sells = [f for f in result.fills if f.order.qty < 0 and f.timestamp is not None]
    if buys:
        fig.add_trace(go.Scatter(x=[f.timestamp for f in buys], y=[f.fill_price for f in buys],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
            hovertemplate="%{x|%Y-%m-%d}<br>Buy @ $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    if sells:
        fig.add_trace(go.Scatter(x=[f.timestamp for f in sells], y=[f.fill_price for f in sells],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
            hovertemplate="%{x|%Y-%m-%d}<br>Sell @ $%{y:.2f}<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data["volume"], name="Volume",
        marker_color="rgba(100,100,200,0.4)", showlegend=False), row=2, col=1)
    fig.update_layout(title=f"{symbol} — trades", xaxis_rangeslider_visible=False,
                      height=540, hovermode="x unified")
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _allocation_fig(alloc: pd.DataFrame) -> go.Figure:
    """Stacked area chart showing dollar allocation per symbol + cash over time."""
    pct = alloc.div(alloc.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for col in pct.columns:
        fig.add_trace(go.Scatter(
            x=pct.index, y=pct[col], name=col, stackgroup="one", mode="lines",
            hovertemplate="%{x|%Y-%m-%d}<br>" + col + ": %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(title="Portfolio Allocation Over Time (%)",
                      xaxis_title="Date", yaxis_title="Allocation (%)",
                      hovermode="x unified", height=360,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def _leaderboard_df(named_metrics: list[tuple[str, dict]], sort_col: str) -> pd.DataFrame:
    rows = []
    for name, m in named_metrics:
        rows.append({
            "Strategy":       name,
            "Total Return %": m["total_return"],
            "Ann. Return %":  m["annualized_return"],
            "Sharpe":         m["sharpe"],
            "Sortino":        m["sortino"],
            "Max DD %":       m["max_drawdown"],
            "Calmar":         m["calmar"],
            "Win Rate %":     m["win_rate"],
        })
    df = pd.DataFrame(rows).sort_values(sort_col, ascending=(sort_col == "Max DD %")).reset_index(drop=True)
    df.index += 1
    return df


def _style_leaderboard(df: pd.DataFrame):
    def color(val, col):
        if col in ("Total Return %", "Ann. Return %", "Sharpe", "Sortino", "Calmar", "Win Rate %"):
            return "color: #26a69a" if val > 0 else "color: #ef5350"
        if col == "Max DD %":
            return "color: #ef5350" if val < -20 else "color: #26a69a"
        return ""
    return df.style.apply(lambda c: [color(v, c.name) for v in c], axis=0).format({
        "Total Return %": "{:.2f}%", "Ann. Return %": "{:.2f}%",
        "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
        "Max DD %": "{:.2f}%", "Calmar": "{:.3f}", "Win Rate %": "{:.1f}%",
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 AI Trading Gym")
    st.divider()

    mode = st.radio("Mode", ["Single Symbol", "Portfolio"], horizontal=True)
    st.divider()

    if mode == "Single Symbol":
        symbol_input = st.text_input("Symbol", value="AAPL").upper().strip()
    else:
        symbols_raw = st.text_input("Symbols (comma-separated)", value="AAPL, MSFT, GOOGL, AMZN")
        symbol_list = [s.strip().upper() for s in symbols_raw.split(",") if s.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start", value=pd.Timestamp("2020-01-01"))
    with col2:
        end = st.date_input("End", value=pd.Timestamp("2024-12-31"))

    if mode == "Single Symbol":
        selected = st.multiselect("Strategies", list(SINGLE_STRATEGIES),
                                  default=list(SINGLE_STRATEGIES))
    else:
        selected = st.multiselect("Portfolio Strategies", list(PORTFOLIO_STRATEGIES),
                                  default=list(PORTFOLIO_STRATEGIES))

    initial_cash = st.number_input("Initial Cash ($)", value=100_000, step=10_000)
    sort_by = st.selectbox("Rank by", list(SORT_COL))
    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)


# ── Landing page ──────────────────────────────────────────────────────────────

if not run_btn:
    st.title("AI Trading Gym")
    st.markdown("Configure a backtest in the sidebar and click **▶ Run Backtest**.")
    st.markdown("""
**Modes**
- **Single Symbol** — compare multiple strategies on one ticker, ranked leaderboard
- **Portfolio** — compare portfolio strategies that allocate shared capital across multiple tickers simultaneously
    """)
    st.stop()

sort_col = SORT_COL[sort_by]
start_str, end_str = str(start), str(end)


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE SYMBOL MODE
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Single Symbol":
    if not selected:
        st.warning("Select at least one strategy.")
        st.stop()

    with st.spinner(f"Backtesting {len(selected)} strategies on {symbol_input}..."):
        data = fetch(symbol_input, start_str, end_str)
        engine = BacktestEngine(initial_cash=initial_cash)
        results = [(name, engine.run(SINGLE_STRATEGIES[name](), data, symbol_input))
                   for name in selected]

    named_metrics = [(name, r.metrics) for name, r in results]
    board = _leaderboard_df(named_metrics, sort_col)

    st.title(f"📊 {symbol_input}  ·  {start} → {end}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Sharpe",    f"{board['Sharpe'].iloc[0]:.2f}",        board["Strategy"].iloc[0])
    c2.metric("Best Return",    f"{board['Total Return %'].iloc[0]:.1f}%", board["Strategy"].iloc[0])
    c3.metric("Bars",           f"{len(data):,}",                          f"{data.index[0].date()} → {data.index[-1].date()}")
    c4.metric("Strategies",     str(len(results)),                         "compared")
    st.divider()

    tab_board, tab_equity, tab_dd, tab_trades = st.tabs(
        ["🏆 Leaderboard", "📈 Equity Curves", "📉 Drawdown", "🔍 Trade Detail"])

    with tab_board:
        st.dataframe(_style_leaderboard(board), use_container_width=True, height=250)

    with tab_equity:
        curves = {name: r.equity_curve for name, r in results}
        st.plotly_chart(_equity_fig(curves, f"Equity Curves — {symbol_input}"), use_container_width=True)

    with tab_dd:
        st.plotly_chart(_drawdown_fig({name: r.equity_curve for name, r in results}), use_container_width=True)

    with tab_trades:
        detail_name = st.selectbox("Strategy", [n for n, _ in results])
        detail_result = next(r for n, r in results if n == detail_name)
        st.plotly_chart(_price_trades_fig(data, symbol_input, detail_result), use_container_width=True)
        fills = detail_result.fills
        if fills:
            st.dataframe(pd.DataFrame([{
                "Date":  f.timestamp.date() if f.timestamp else "—",
                "Side":  "BUY" if f.order.qty > 0 else "SELL",
                "Qty":   int(abs(f.order.qty)),
                "Price": f"${f.fill_price:.2f}",
                "Value": f"${abs(f.order.qty) * f.fill_price:,.0f}",
            } for f in fills]), use_container_width=True, height=220)
        else:
            st.info("No trades executed.")


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO MODE
# ══════════════════════════════════════════════════════════════════════════════

else:
    if len(symbol_list) < 2:
        st.warning("Enter at least 2 symbols for portfolio mode.")
        st.stop()
    if not selected:
        st.warning("Select at least one portfolio strategy.")
        st.stop()

    with st.spinner(f"Fetching data for {', '.join(symbol_list)}..."):
        all_data = {sym: fetch(sym, start_str, end_str) for sym in symbol_list}

    with st.spinner(f"Running {len(selected)} portfolio strategies..."):
        engine = PortfolioBacktestEngine(initial_cash=initial_cash)
        port_results: list[tuple[str, PortfolioBacktestResult]] = [
            (name, engine.run(PORTFOLIO_STRATEGIES[name](), all_data))
            for name in selected
        ]

    # Add individual buy-and-hold benchmarks for comparison
    single_engine = BacktestEngine(initial_cash=initial_cash / len(symbol_list))
    bh_curves = {}
    for sym in symbol_list:
        r = single_engine.run(BuyAndHold(), all_data[sym], sym)
        bh_curves[f"B&H {sym}"] = r.equity_curve * len(symbol_list)  # scale to same cash base

    named_metrics = [(name, r.metrics) for name, r in port_results]
    board = _leaderboard_df(named_metrics, sort_col)

    symbols_str = ", ".join(symbol_list)
    st.title(f"📊 Portfolio: {symbols_str}  ·  {start} → {end}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Sharpe",  f"{board['Sharpe'].iloc[0]:.2f}",         board["Strategy"].iloc[0])
    c2.metric("Best Return",  f"{board['Total Return %'].iloc[0]:.1f}%", board["Strategy"].iloc[0])
    c3.metric("Symbols",      str(len(symbol_list)),                      symbols_str[:30])
    c4.metric("Strategies",   str(len(port_results)),                     "compared")
    st.divider()

    tab_board, tab_equity, tab_alloc, tab_dd, tab_trades = st.tabs(
        ["🏆 Leaderboard", "📈 Equity Curves", "🥧 Allocation", "📉 Drawdown", "🔍 Trade Detail"])

    with tab_board:
        st.dataframe(_style_leaderboard(board), use_container_width=True, height=250)

    with tab_equity:
        curves = {name: r.equity_curve for name, r in port_results}
        curves.update(bh_curves)  # overlay individual B&H benchmarks
        st.plotly_chart(_equity_fig(curves, "Portfolio Equity vs Individual Buy & Hold"), use_container_width=True)

    with tab_alloc:
        alloc_name = st.selectbox("Strategy", [n for n, _ in port_results], key="alloc_sel")
        alloc_result = next(r for n, r in port_results if n == alloc_name)
        st.plotly_chart(_allocation_fig(alloc_result.allocation_history), use_container_width=True)

    with tab_dd:
        st.plotly_chart(_drawdown_fig({name: r.equity_curve for name, r in port_results}), use_container_width=True)

    with tab_trades:
        trade_name = st.selectbox("Strategy", [n for n, _ in port_results], key="trade_sel")
        trade_result = next(r for n, r in port_results if n == trade_name)

        fills = trade_result.fills
        if fills:
            fill_rows = [{
                "Date":   f.timestamp.date() if f.timestamp else "—",
                "Symbol": f.order.symbol,
                "Side":   "BUY" if f.order.qty > 0 else "SELL",
                "Qty":    int(abs(f.order.qty)),
                "Price":  f"${f.fill_price:.2f}",
                "Value":  f"${abs(f.order.qty) * f.fill_price:,.0f}",
            } for f in fills]
            st.dataframe(pd.DataFrame(fill_rows), use_container_width=True, height=300)

            # Mini price chart per symbol with trade markers
            sym_sel = st.selectbox("Price chart for", symbol_list)
            sym_fills_result = BacktestResult(
                strategy_name=trade_name, symbol=sym_sel,
                equity_curve=trade_result.equity_curve,
                fills=[f for f in fills if f.order.symbol == sym_sel],
            )
            st.plotly_chart(_price_trades_fig(all_data[sym_sel], sym_sel, sym_fills_result),
                            use_container_width=True)
        else:
            st.info("No trades executed.")
