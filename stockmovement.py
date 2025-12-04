# app.py – FINAL POLISHED & NARRATED VERSION

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="StockLab Pro", layout="wide")
sns.set_style("darkgrid")

st.title("StockLab Pro – Your Personal Quant Dashboard")
st.markdown("One ticker → all the math that matters, explained in plain English")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
period = st.sidebar.selectbox("Lookback", ["1y", "2y", "5y", "10y", "max"], index=0)

@st.cache_data(show_spinner="Downloading price history...")
def get_data(ticker, period):
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if data.empty:
        st.error("No data found — check the ticker symbol")
        st.stop()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    price = data['Close'] if 'Close' in data.columns else data.iloc[:,0]
    df = pd.DataFrame({"Price": price.dropna()})
    df["Return"] = np.log(df["Price"] / df["Price"].shift(1))
    df = df.dropna()
    return df

df = get_data(ticker, period)

# Top metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${df['Price'].iloc[-1]:.2f}")
c2.metric("Total Return", f"{(df['Price'].iloc[-1]/df['Price'].iloc[0]-1)*100:+.1f}%")
c3.metric("Annual Volatility", f"{df['Return'].std()*np.sqrt(252)*100:.1f}%")
sharpe = (df['Return'].mean()*252 - 0.04)/(df['Return'].std()*np.sqrt(252) + 1e-8)
c4.metric("Sharpe Ratio (rf=4%)", f"{sharpe:.2f}")

# ——————————— SECTION 0: Price Chart ———————————
st.subheader(f"Price History – {ticker}")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df.index, df['Price'], color="#1f77b4", linewidth=2)
ax.set_ylabel("Price ($)", fontsize=12)
ax.set_title(f"{ticker} – Adjusted Close Price", fontsize=14)
st.pyplot(fig)
st.info("This is simply what happened. Everything below tries to explain *why* and *what might happen next*.")

# ——————————— SECTION 1: Random Walk Test ———————————
st.subheader("1. Does the stock behave like a drunk random walk?")
adf = adfuller(df['Return'])
p = adf[1]
if p < 0.05:
    st.success("Yes — daily moves are unpredictable (classic random walk behavior)")
else:
    st.warning("No strong evidence of pure randomness — there might be momentum or mean reversion")
st.caption("Most stocks pass this test. If they didn’t, day-traders would be rich.")

# ——————————— SECTION 2: GBM Parameters ———————————
st.subheader("2. The ‘uphill drunk guy’ model (Geometric Brownian Motion)")
mu = df['Return'].mean() * 252
sigma = df['Return'].std() * np.sqrt(252)
st.write(f"Annual drift (average uphill speed): **{mu*100:+.2f}% per year**")
st.write(f"Annual volatility (drunken wobble size): **{sigma*100:.1f}%**")
if mu > 0.07:
    st.success("Strong positive drift — the hill is noticeably uphill")
elif mu > 0:
    st.info("Gentle uphill slope")
else:
    st.warning("Flat or downhill — rare for stocks over long periods")

# ——————————— SECTION 3: Monte Carlo Forecast (now perfectly labeled + story) ———————————
st.subheader("3. Where could the price be in 12 months? (1,000 realistic paths)")

days, n = 252, 1000
price0 = df['Price'].iloc[-1]
drift_daily = (mu - 0.5*sigma**2)/252
vol_daily = sigma/np.sqrt(252)
rand = np.random.randn(days, n)
log_rets = drift_daily + vol_daily * rand
paths = price0 * np.cumprod(np.exp(log_rets), axis=0)
paths = np.vstack([np.full(n, price0), paths])

fig, ax = plt.subplots(figsize=(13,7))
ax.plot(paths[:, :60], color="steelblue", linewidth=0.9, alpha=0.7)
ax.plot(paths.mean(axis=1), color="red", linewidth=3.5, label="Average future path")
ax.axhline(price0, color="black", linestyle="--", linewidth=2, label="Today’s price")
ax.set_xlabel("Trading Days into the Future", fontsize=12)
ax.set_ylabel("Price ($)", fontsize=12)
ax.set_title(f"{ticker} – 1,000 Possible 12-Month Futures (GBM simulation)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
st.pyplot(fig)

p10, p50, p90 = np.percentile(paths[-1], [10, 50, 90])
st.markdown(f"""
**In plain English:**  
- There’s a 10% chance the stock ends the year **below ${p10:.0f}** (bad luck, big storms)  
- The **most likely outcome** (median path) is around **${p50:.0f}**  
- There’s a 10% chance it finishes **above ${p90:.0f}** (everything goes right)  
This is the same math used to price options on Wall Street.
""")

# ——————————— SECTION 4: GARCH ———————————
st.subheader("4. Volatility clustering – ‘When it rains, it pours’")
model = arch_model(df['Return']*100, vol='Garch', p=1, q=1, dist='Normal')
res = model.fit(disp='off')

garch_table = pd.DataFrame({
    "Parameter": ["ω (baseline vol)", "α (news impact)", "β (persistence)"],
    "Estimate": [res.params[0], res.params[1], res.params[2]],
    "p-value": [res.pvalues[0], res.pvalues[1], res.pvalues[2]]
})
st.table(garch_table.style.format({"Estimate": "{:.6f}", "p-value": "{:.4f}"}))

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,8))
ax1.plot(df.index, abs(df['Return']*100), color="gray")
ax1.set_title("Absolute daily moves — notice the clusters of big swings?")
ax2.plot(res.conditional_volatility.index, res.conditional_volatility, color="purple", linewidth=2)
ax2.set_title("GARCH(1,1) estimated volatility — calm periods stay calm, stormy periods stay stormy")
st.pyplot(fig)

st.info("This is the famous ‘volatility clustering’ effect. One bad day rarely travels alone.")

# ——————————— SECTION 5: Mean Reversion ———————————
st.subheader("5. Does the stock try to return to an ‘average’ price?")
log_price = np.log(df['Price'])
d_log = log_price.diff().dropna()
log_lag = log_price.shift(1).loc[d_log.index]

if len(d_log) > 50:
    X = sm.add_constant(log_lag.values[:-1])
    y = d_log.values[1:]
    ou = sm.OLS(y, X).fit()
    lambda_ = ou.params[1]
    kappa = -np.log(lambda_) * 252 if 0 < lambda_ < 1 else 0
    half_life = f"{np.log(2)/kappa:.0f} days" if kappa > 0.1 else "Very weak or none"
else:
    half_life = "Not enough data"

st.write(f"Estimated mean-reversion half-life: **{half_life}**")
if "days" in half_life and int(half_life.split()[0]) < 200:
    st.success("Some mean reversion exists — big deviations tend to fade")
else:
    st.warning("Trend dominates — no strong rubber-band effect")

# ——————————— SECTION 6: Autocorrelation ———————————
st.subheader("6. Are moves predictable day-to-day?")
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))
sm.graphics.tsa.plot_acf(df['Return'], lags=30, ax=ax1, title="Raw Returns → Should be flat (no pattern)")
sm.graphics.tsa.plot_acf(df['Return']**2, lags=30, ax=ax2, title="Squared Returns → Spikes = volatility clustering")
st.pyplot(fig)

st.info("Left chart flat = good (no obvious predictability). Right chart spiky = classic GARCH signal.")

# Final celebration
st.success("All professional-grade tests completed — you now understand this stock better than 99% of investors!")
st.balloons()
st.caption("Built with love using Streamlit • yfinance • arch • statsmodels")
