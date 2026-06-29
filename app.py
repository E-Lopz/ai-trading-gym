"""
Streamlit visualizer for strategy backtests and comparisons.

Run with:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.fetchers import YFinanceFetcher
from backtesting.engine import BacktestEngine
from backtesting.results import BacktestResult
from strategies.buy_and_hold import BuyAndHold
from strategies.sma_crossover import SMACrossover
from strategies.rsi import RSIMeanReversion

st.set_page_config(page_title="AI Trading Gym", layout="wide", page_icon="📈")

STRATEGY_OPTIONS = {
    "Buy & Hold":            lambda: BuyAndHold(),
    "SMA Crossover (10/30)": lambda: SMACrossover(fast=10, slow=30),
    "SMA Crossover (20/50)": lambda: SMACrossover(fast=20, slow=50),
    "RSI Mean Reversion":    lambda: RSIMeanReversion(period=14, oversold=30, overbought=70),
    "RSI (fast, tight)":     lambda: RSIMeanReversion(period=7,  oversold=25, overbought=75),
}

METRIC_DISPLAY = {
    "total_return":      ("Total Return",     "%"),
    "annualized_return": ("Ann. Return",      "%"),
    "sharpe":            ("Sharpe Ratio",     ""),
    "sortino":           ("Sortino Ratio",    ""),
    "max_drawdown":      ("Max Drawdown",     "%"),
    "calmar":            ("Calmar Ratio",     ""),
    "win_rate":          ("Win Rate",         "%"),
}


# ── Sidebar controls ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 AI Trading Gym")
    st.divider()

    symbol = st.text_input("Symbol", value="AAPL").upper().strip()

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input("Start", value=pd.Timestamp("2020-01-01"))
    with col2:
        end = st.date_input("End", value=pd.Timestamp("2024-12-31"))

    selected_strategies = st.multiselect(
        "Strategies",
        options=list(STRATEGY_OPTIONS.keys()),
        default=list(STRATEGY_OPTIONS.keys()),
    )

    initial_cash = st.number_input("Initial Cash ($)", value=100_000, step=10_000)
    sort_by = st.selectbox("Rank by", ["Sharpe Ratio", "Total Return", "Ann. Return", "Sortino Ratio", "Calmar Ratio"])

    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)

sort_col_map = {
    "Sharpe Ratio":   "sharpe",
    "Total Return":   "total_return",
    "Ann. Return":    "annualized_return",
    "Sortino Ratio":  "sortino",
    "Calmar Ratio":   "calmar",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching market data...")
def fetch_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    return YFinanceFetcher().fetch(symbol, start=start, end=end, interval="1d", cache=False)


def run_backtests(data: pd.DataFrame, strategy_names: list, cash: float) -> list[tuple[str, BacktestResult]]:
    engine = BacktestEngine(initial_cash=cash)
    results = []
    for name in strategy_names:
        strategy = STRATEGY_OPTIONS[name]()
        result = engine.run(strategy, data, symbol)
        results.append((name, result))
    return results


def build_leaderboard(results: list, sort_key: str) -> pd.DataFrame:
    rows = []
    for name, r in results:
        m = r.metrics
        rows.append({
            "Strategy":       name,
            "Total Return %": m["total_return"],
            "Ann. Return %":  m["annualized_return"],
            "Sharpe":         m["sharpe"],
            "Sortino":        m["sortino"],
            "Max DD %":       m["max_drawdown"],
            "Calmar":         m["calmar"],
            "Win Rate %":     m["win_rate"],
            "# Trades":       len([f for f in r.fills if f.order.qty > 0]),
        })
    df = pd.DataFrame(rows)
    col_map = {"total_return": "Total Return %", "annualized_return": "Ann. Return %",
               "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar"}
    df = df.sort_values(col_map.get(sort_key, "Sharpe"), ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def equity_chart(results: list) -> go.Figure:
    fig = go.Figure()
    for name, r in results:
        fig.add_trace(go.Scatter(
            x=r.equity_curve.index,
            y=r.equity_curve.values,
            name=name,
            mode="lines",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title="Equity Curves",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


def price_trades_chart(data: pd.DataFrame, name: str, result: BacktestResult) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.04)

    fig.add_trace(go.Candlestick(
        x=data.index, open=data["open"], high=data["high"],
        low=data["low"], close=data["close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        showlegend=False,
    ), row=1, col=1)

    buys  = [f for f in result.fills if f.order.qty > 0 and f.timestamp is not None]
    sells = [f for f in result.fills if f.order.qty < 0 and f.timestamp is not None]

    if buys:
        fig.add_trace(go.Scatter(
            x=[f.timestamp for f in buys],
            y=[f.fill_price for f in buys],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
            hovertemplate="%{x|%Y-%m-%d}<br>Buy @ $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    if sells:
        fig.add_trace(go.Scatter(
            x=[f.timestamp for f in sells],
            y=[f.fill_price for f in sells],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
            hovertemplate="%{x|%Y-%m-%d}<br>Sell @ $%{y:.2f}<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=data.index, y=data["volume"], name="Volume",
        marker_color="rgba(100,100,200,0.4)", showlegend=False,
    ), row=2, col=1)

    fig.update_layout(
        title=f"{name} — {symbol} trades",
        xaxis_rangeslider_visible=False,
        height=540,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def drawdown_chart(results: list) -> go.Figure:
    fig = go.Figure()
    for name, r in results:
        eq = r.equity_curve
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name, mode="lines", fill="tozeroy",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.1f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    return fig


# ── Main content ──────────────────────────────────────────────────────────────

if not run_btn:
    st.title("AI Trading Gym")
    st.markdown(
        "Configure a backtest in the sidebar and click **▶ Run Backtest** to compare strategies."
    )
    st.markdown("""
**What you can do:**
- Compare multiple strategies on the same symbol and date range
- See equity curves, drawdowns, and trade markers on the price chart
- Rank by Sharpe, return, Sortino, or Calmar ratio
    """)
    st.stop()

if not selected_strategies:
    st.warning("Select at least one strategy in the sidebar.")
    st.stop()

# Run
with st.spinner(f"Backtesting {len(selected_strategies)} strategies on {symbol}..."):
    data = fetch_data(symbol, str(start), str(end))
    results = run_backtests(data, selected_strategies, initial_cash)

sort_key = sort_col_map.get(sort_by, "sharpe")
leaderboard = build_leaderboard(results, sort_key)
best_name, best_result = results[0]  # default detail view to first strategy

# ── Header KPIs ──────────────────────────────────────────────────────────────
st.title(f"📊 {symbol}  ·  {start} → {end}")

best_metrics = best_result.metrics
kpi_cols = st.columns(5)
kpi_data = [
    ("Best Sharpe",    f"{leaderboard['Sharpe'].iloc[0]:.2f}",     leaderboard['Strategy'].iloc[0]),
    ("Best Return",    f"{leaderboard['Total Return %'].iloc[0]:.1f}%", leaderboard['Strategy'].iloc[0]),
    ("Best Ann. Ret.", f"{leaderboard['Ann. Return %'].iloc[0]:.1f}%", leaderboard['Strategy'].iloc[0]),
    ("Bars",           f"{len(data):,}",                           f"{data.index[0].date()} → {data.index[-1].date()}"),
    ("Strategies",     str(len(results)),                          "compared"),
]
for col, (label, value, sub) in zip(kpi_cols, kpi_data):
    col.metric(label, value, sub)

st.divider()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_board, tab_equity, tab_dd, tab_trades = st.tabs(
    ["🏆 Leaderboard", "📈 Equity Curves", "📉 Drawdown", "🔍 Trade Detail"]
)

with tab_board:
    def color_metric(val, col):
        if col in ("Total Return %", "Ann. Return %", "Sharpe", "Sortino", "Calmar", "Win Rate %"):
            color = "#26a69a" if val > 0 else "#ef5350"
        elif col == "Max DD %":
            color = "#ef5350" if val < -20 else "#26a69a"
        else:
            return ""
        return f"color: {color}"

    styled = leaderboard.style.apply(
        lambda col: [color_metric(v, col.name) for v in col], axis=0
    ).format({
        "Total Return %": "{:.2f}%",
        "Ann. Return %":  "{:.2f}%",
        "Sharpe":         "{:.3f}",
        "Sortino":        "{:.3f}",
        "Max DD %":       "{:.2f}%",
        "Calmar":         "{:.3f}",
        "Win Rate %":     "{:.1f}%",
    })
    st.dataframe(styled, use_container_width=True, height=250)

with tab_equity:
    st.plotly_chart(equity_chart(results), use_container_width=True)

with tab_dd:
    st.plotly_chart(drawdown_chart(results), use_container_width=True)

with tab_trades:
    detail_name = st.selectbox("Strategy", [n for n, _ in results])
    detail_result = next(r for n, r in results if n == detail_name)
    st.plotly_chart(price_trades_chart(data, detail_name, detail_result), use_container_width=True)

    fills = detail_result.fills
    if fills:
        fill_rows = [{
            "Date":       f.timestamp.date() if f.timestamp else "—",
            "Side":       "BUY" if f.order.qty > 0 else "SELL",
            "Qty":        int(abs(f.order.qty)),
            "Fill Price": f"${f.fill_price:.2f}",
            "Value":      f"${abs(f.order.qty) * f.fill_price:,.0f}",
        } for f in fills]
        st.dataframe(pd.DataFrame(fill_rows), use_container_width=True, height=220)
    else:
        st.info("No trades executed by this strategy.")
