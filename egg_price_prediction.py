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
# 1) Static + optional live-fetch up to March 2025
# ------------------------------------------------------------------------------

BLS_API_URL    = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
EGGS_SERIES_ID = "APU000071761"  # “eggs, retail, per dozen”
API_KEY        = os.getenv("BLS_API_KEY", None)

STATIC_DATA = np.array([
    [2017,  1, 1.47], [2017,  2, 1.48], [2017,  3, 1.49], [2017,  4, 1.50],
    [2017,  5, 1.51], [2017,  6, 1.52], [2017,  7, 1.53], [2017,  8, 1.54],
    [2017,  9, 1.55], [2017, 10, 1.56], [2017, 11, 1.57], [2017, 12, 1.58],
    [2018,  1, 1.74], [2018,  2, 1.75], [2018,  3, 1.76], [2018,  4, 1.77],
    [2018,  5, 1.78], [2018,  6, 1.79], [2018,  7, 1.80], [2018,  8, 1.81],
    [2018,  9, 1.82], [2018, 10, 1.83], [2018, 11, 1.84], [2018, 12, 1.85],
    [2019,  1, 1.40], [2019,  2, 1.41], [2019,  3, 1.42], [2019,  4, 1.43],
    [2019,  5, 1.44], [2019,  6, 1.45], [2019,  7, 1.46], [2019,  8, 1.47],
    [2019,  9, 1.48], [2019, 10, 1.49], [2019, 11, 1.50], [2019, 12, 1.51],
    [2020,  1, 1.461], [2020,  2, 1.449], [2020,  3, 1.525], [2020,  4, 2.019],
    [2020,  5, 1.640], [2020,  6, 1.554], [2020,  7, 1.401], [2020,  8, 1.328],
    [2020,  9, 1.353], [2020, 10, 1.408], [2020, 11, 1.450], [2020, 12, 1.481],
    [2021,  1, 1.466], [2021,  2, 1.597], [2021,  3, 1.625], [2021,  4, 1.620],
    [2021,  5, 1.625], [2021,  6, 1.642], [2021,  7, 1.642], [2021,  8, 1.709],
    [2021,  9, 1.835], [2021, 10, 1.821], [2021, 11, 1.718], [2021, 12, 1.788],
    [2022,  1, 1.929], [2022,  2, 2.005], [2022,  3, 2.046], [2022,  4, 2.520],
    [2022,  5, 2.863], [2022,  6, 2.707], [2022,  7, 2.936], [2022,  8, 3.116],
    [2022,  9, 2.902], [2022, 10, 3.419], [2022, 11, 3.589], [2022, 12, 4.250],
    [2023,  1, 4.823], [2023,  2, 4.211], [2023,  3, 3.446], [2023,  4, 3.270],
    [2023,  5, 2.666], [2023,  6, 2.219], [2023,  7, 2.094], [2023,  8, 2.043],
    [2023,  9, 2.065], [2023, 10, 2.072], [2023, 11, 2.138], [2023, 12, 2.507],
    [2024,  1, 2.522], [2024,  2, 2.996], [2024,  3, 2.992], [2024,  4, 2.864],
    [2024,  5, 2.699], [2024,  6, 2.715], [2024,  7, 3.080], [2024,  8, 3.204],
    [2024,  9, 3.821], [2024, 10, 3.370], [2024, 11, 3.649], [2024, 12, 4.146],
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
        df = pd.concat([df_static, df_dyn], ignore_index=True)
        df = df.drop_duplicates(subset="date", keep="last")
    else:
        df = df_static

    df = df.sort_values("date").reset_index(drop=True)
    return df, ["Year","Month"]

# ------------------------------------------------------------------------------
# 2) Walk-forward + bootstrap CI
# ------------------------------------------------------------------------------

def walk_forward_predict(df, features, n_bootstrap, xgb_params):
    history = df.iloc[:3].copy()
    results = []

    for idx in range(3, len(df)):
        train = history.copy()
        test  = df.iloc[idx]
        X_test = test[features].values.reshape(1, -1)

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
        ax.plot(df["date"], df["retail_price"], color="gray", label="Historical")
        sub = preds.iloc[:i+1]
        ax.fill_between(sub["date"], sub["ci_lower"], sub["ci_upper"], color="lightgray", label="95% CI")
        ax.plot(sub["date"], sub["predicted"], "C1--", label="Predicted")
        ax.plot(sub["date"], sub["actual"], "C0-", linewidth=2, label="Actual")
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
    fig.add_trace(go.Scatter(x=df["date"], y=df["retail_price"], mode="lines", name="Historical", line=dict(color="gray")))
    fig.add_trace(go.Scatter(x=preds["date"], y=preds["ci_upper"], line=dict(color="lightgray"), name="Upper CI"))
    fig.add_trace(go.Scatter(x=preds["date"], y=preds["ci_lower"], line=dict(color="lightgray"), fill="tonexty", name="Lower CI"))
    fig.add_trace(go.Scatter(x=preds["date"], y=preds["predicted"], mode="lines+markers", name="Predicted"))
    fig.add_trace(go.Scatter(x=preds["date"], y=preds["actual"], mode="lines+markers", name="Actual"))
    fig.update_layout(title="Egg Price Forecast (walk-forward)", xaxis_title="Date", yaxis_title="$/dozen",
                      legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig

# ------------------------------------------------------------------------------
# 4) Streamlit UI
# ------------------------------------------------------------------------------

def main():
    st.title("🥚 Egg Price Walk-Forward Forecast")
    df, features = prepare_data()

    last = df.iloc[-1]
    st.metric("Latest reported price (per dozen)", f"${last['retail_price']:.2f}")

    st.sidebar.header("Model hyperparameters")
    n_bootstrap = st.sidebar.slider("Bootstrap samples", 10, 200, 50, step=10)
    xgb_params = {
        "n_estimators":  st.sidebar.slider("n_estimators", 50, 500, 300, step=50),
        "learning_rate": st.sidebar.slider("learning_rate", 0.01, 0.30, 0.10, step=0.01),
        "max_depth":     st.sidebar.slider("max_depth",       2, 10,   5),
        "verbosity":     0,
        "objective":     "reg:squarederror",
    }

    if st.button("Run walk-forward forecast"):
        with st.spinner("Training & forecasting…"):
            preds = walk_forward_predict(df, features, n_bootstrap, xgb_params)

        coverage = ((preds.actual >= preds.ci_lower) &
                    (preds.actual <= preds.ci_upper)).mean()
        mse      = mean_squared_error(preds.actual, preds.predicted)
        rmse     = np.sqrt(mse)
        mae      = mean_absolute_error(preds.actual, preds.predicted)
        mape     = (np.abs((preds.actual - preds.predicted) / preds.actual).mean() * 100)

        st.success(f"Done! 95% CI coverage: {coverage:.1%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.3f}")
        c2.metric("MAE",  f"{mae:.3f}")
        c3.metric("MAPE", f"{mape:.1f}%")

        plot_matplotlib(df, preds)
        st.plotly_chart(plot_plotly(df, preds), use_container_width=True)

if __name__ == "__main__":
    main()

