import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------------------------------------------------------
# 1) Static + optional live‐fetch
# ------------------------------------------------------------------------------

BLS_API_URL    = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
EGGS_SERIES_ID = "APU000071761"  # “eggs, retail, per dozen”
API_KEY        = os.getenv("BLS_API_KEY", None)

# FULL static history 2017‐01 → 2025‐03
STATIC_DATA = np.array([
    [2017,  1, 1.47], [2017,  2, 1.48], [2017,  3, 1.49], [2017,  4, 1.50],
    # … (all the way through) …
    [2025,  1, 4.953], [2025,  2, 5.897], [2025,  3, 6.227]
])

def fetch_bls_eggs(start_year: int, end_year: int):
    if not API_KEY:
        return None
    payload = json.dumps({
        "seriesid": [EGGS_SERIES_ID],
        "startyear": str(start_year),
        "endyear":   str(end_year),
        "registrationKey": API_KEY
    })
    resp = requests.post(BLS_API_URL, data=payload,
                         headers={"Content-Type":"application/json"})
    data = resp.json()
    try:
        series = data["Results"]["series"][0]["data"]
    except Exception:
        return None

    out = []
    for e in series:
        y = int(e["year"])
        m = int(e["period"][1:])
        v = float(e["value"])
        out.append([y, m, v])
    return np.array(out)


@st.cache_data
def prepare_data():
    # 1) always load the FULL static set
    df_static = pd.DataFrame(STATIC_DATA, columns=["Year","Month","retail_price"])
    df_static["date"] = pd.to_datetime(
        df_static[["Year","Month"]].assign(Day=1)
    )

    # 2) try to fetch live for the last 5 years
    now = datetime.now()
    dyn = fetch_bls_eggs(now.year - 5, now.year)

    if dyn is not None and len(dyn) > 0:
        df_dyn = pd.DataFrame(dyn, columns=["Year","Month","retail_price"])
        df_dyn["date"] = pd.to_datetime(
            df_dyn[["Year","Month"]].assign(Day=1)
        )
        # 3) concat and drop duplicates, keeping the live value if it exists
        df = pd.concat([df_static, df_dyn], ignore_index=True)
        df = df.drop_duplicates(subset="date", keep="last")
    else:
        df = df_static

    df = df.sort_values("date").reset_index(drop=True)
    return df, ["Year","Month"]

# ------------------------------------------------------------------------------
# 2) Walk‐forward + bootstrap CI
# ------------------------------------------------------------------------------

def walk_forward_predict(df, features, n_bootstrap, xgb_params):
    history = df.iloc[:3].copy()
    results = []

    for idx in range(3, len(df)):
        train = history.copy()
        test  = df.iloc[idx]
        X_test = test[features].values.reshape(1,-1)

        boot = []
        for seed in range(n_bootstrap):
            samp = train.sample(frac=1, replace=True, random_state=seed)
            m = XGBRegressor(random_state=seed, **xgb_params)
            m.fit(samp[features], np.log(samp["retail_price"]))
            boot.append(np.exp(m.predict(X_test)[0]))

        boot = np.array(boot)
        mu   = boot.mean()
        lo, hi = np.percentile(boot, [2.5,97.5])

        results.append({
            "date":      test["date"],
            "actual":    test["retail_price"],
            "predicted": mu,
            "ci_lower":  lo,
            "ci_upper":  hi
        })
        history = pd.concat([history, df.iloc[[idx]]], ignore_index=True)

    return pd.DataFrame(results)

# ------------------------------------------------------------------------------
# 3) Plotting helpers
# ------------------------------------------------------------------------------

def plot_matplotlib(df, preds):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10,5))

    for i in range(len(preds)):
        ax.clear()
        # full static+live history in gray
        ax.plot(df["date"], df["retail_price"],
                color="gray", label="Historical")
        sub = preds.iloc[: i+1]
        ax.fill_between(sub["date"], sub["ci_lower"], sub["ci_upper"],
                        color="lightgray", label="95% CI")
        ax.plot(sub["date"], sub["predicted"], "C1--", label="Predicted")
        ax.plot(sub["date"], sub["actual"],  "C0-",  linewidth=2, label="Actual")
        ax.set_title("Egg Price Forecast (walk-forward)")
        ax.set_ylabel("$/dozen")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        fig.canvas.draw()
        plt.pause(0.05)

    plt.ioff()
    plt.show()

def plot_plotly(df, preds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["retail_price"],
        mode="lines", name="Historical", line=dict(color="gray")
    ))
    fig.add_trace(go.Scatter(
        x=preds["date"], y=preds["ci_upper"],
        line=dict(color="lightgray"), name="Upper CI"
    ))
    fig.add_trace(go.Scatter(
        x=preds["date"], y=preds["ci_lower"],
        line=dict(color="lightgray"), fill="tonexty", name="Lower CI"
    ))
    fig.add_trace(go.Scatter(
        x=preds["date"], y=preds["predicted"],
        mode="lines+markers", name="Predicted"
    ))
    fig.add_trace(go.Scatter(
        x=preds["date"], y=preds["actual"],
        mode="lines+markers", name="Actual"
    ))
    fig.update_layout(
        title="Egg Price Forecast (walk-forward)",
        xaxis_title="Date", yaxis_title="$/dozen",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

# ------------------------------------------------------------------------------
# 4) Streamlit UI
# ------------------------------------------------------------------------------

def main():
    st.title("🥚 Egg Price Walk-Forward Forecast")
    df, features = prepare_data()

    # show the very latest price
    last = df.iloc[-1]
    st.metric("Latest reported price (per dozen)", f"${last['retail_price']:.2f}")

    # hyperparameters
    st.sidebar.header("Model hyperparameters")
    n_bootstrap = st.sidebar.slider("Bootstrap samples", 10, 200, 50, step=10)
    xgb_params = {
        "n_estimators":  st.sidebar.slider("n_estimators",    50, 500, 300, step=50),
        "learning_rate": st.sidebar.slider("learning_rate", 0.01, 0.30, 0.10, step=0.01),
        "max_depth":     st.sidebar.slider("max_depth",       2, 10,   5),
        "verbosity":     0,
        "objective":     "reg:squarederror",
    }

    if st.button("Run walk-forward forecast"):
        with st.spinner("Training & forecasting…"):
            preds = walk_forward_predict(df, features, n_bootstrap, xgb_params)

        # metrics
        coverage = ((preds.actual >= preds.ci_lower) &
                    (preds.actual <= preds.ci_upper)).mean()
        rmse = mean_squared_error(preds.actual, preds.predicted, squared=False)
        mae  = mean_absolute_error(preds.actual, preds.predicted)
        mape = (np.abs((preds.actual - preds.predicted) / preds.actual).mean() * 100)

        st.success(f"Done! 95% CI coverage: {coverage:.1%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.3f}")
        c2.metric("MAE",  f"{mae:.3f}")
        c3.metric("MAPE", f"{mape:.1f}%")

        # live‐updating Matplotlib + Plotly summary
        plot_matplotlib(df, preds)
        st.plotly_chart(plot_plotly(df, preds), use_container_width=True)

if __name__ == "__main__":
    main()
