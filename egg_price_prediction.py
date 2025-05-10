import os
import json
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------------------------------------------------
# 1) Static + optional live‐fetch
# -------------------------------------------------------------------
BLS_API_URL    = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
EGGS_SERIES_ID = "APU000071761"
API_KEY        = os.getenv("BLS_API_KEY", None)

STATIC_DATA = np.array([
    [2017,  1, 1.47], [2017,  2, 1.48], [2017,  3, 1.49], [2017,  4, 1.50],
    # … (all the way through) …
    [2021, 11, 1.718], [2021, 12, 1.788],
])

@st.cache_data
def prepare_data():
    df = pd.DataFrame(STATIC_DATA, columns=["Year","Month","retail_price"])
    df["date"] = pd.to_datetime(df[["Year","Month"]].assign(Day=1))
    features = ["Year","Month"]
    return df, features

def walk_forward_predict(df, features, n_bootstrap=50, xgb_params=None):
    history = df.iloc[:3].copy()
    results = []

    for idx in range(3, len(df)):
        train = history
        test_row = df.iloc[idx]
        X_test = test_row[features].values.reshape(1, -1)

        boot_preds = []
        for seed in range(n_bootstrap):
            sample = train.sample(frac=1.0, replace=True, random_state=seed)
            params = xgb_params or {}
            m = XGBRegressor(**params, random_state=seed)
            m.fit(sample[features], np.log(sample["retail_price"]))
            boot_preds.append(np.exp(m.predict(X_test)[0]))

        boot_preds = np.array(boot_preds)
        pred_mean = boot_preds.mean()
        ci_low, ci_high = np.percentile(boot_preds, [2.5, 97.5])

        results.append({
            "date":     test_row["date"],
            "actual":   test_row["retail_price"],
            "predicted":pred_mean,
            "ci_lower": ci_low,
            "ci_upper": ci_high
        })

        history = pd.concat([history, df.iloc[[idx]]], ignore_index=True)

    return pd.DataFrame(results)

def plot_matplotlib(df, preds):
    # simple static summary plot
    plt.figure(figsize=(8,4))
    plt.fill_between(preds.date, preds.ci_lower, preds.ci_upper, color="lightgray")
    plt.plot(preds.date, preds.predicted, "r--", label="Predicted")
    plt.plot(preds.date, preds.actual, "b-", lw=2, label="Actual")
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())

def plot_plotly(df, preds):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds.date, y=preds.actual, name="Actual", mode="lines"))
    fig.add_trace(go.Scatter(x=preds.date, y=preds.predicted, name="Predicted", mode="lines"))
    fig.add_trace(go.Scatter(
        x=pd.concat([preds.date, preds.date[::-1]]),
        y=pd.concat([preds.ci_upper, preds.ci_lower[::-1]]),
        fill="toself", name="95% CI", line=dict(color="lightgray"), showlegend=False
    ))
    return fig

def main():
    st.title("Egg Price Prediction (Walk-Forward)")
    df, features = prepare_data()

    st.write("Historical data sample:")
    st.dataframe(df.head())

    st.markdown("## Forecast Settings")
    n_bootstrap = st.slider("Bootstrap ensemble size", 10, 200, 50)

    xgb_params = {
        "n_estimators": st.number_input("n_estimators", 50, 500, 300, step=50),
        "learning_rate": st.number_input("learning_rate", 0.01, 0.5, 0.1, step=0.01),
        "max_depth":     st.number_input("max_depth", 1, 10, 5)
    }
    xgb_params["verbosity"]  = 0
    xgb_params["objective"]  = "reg:squarederror"

    if st.button("Run walk-forward forecast"):
        with st.spinner("Training & forecasting…"):
            preds = walk_forward_predict(df, features, n_bootstrap, xgb_params)

        # metrics
        coverage = ((preds.actual >= preds.ci_lower) &
                    (preds.actual <= preds.ci_upper)).mean()

        # ← replace single‐liner RMSE with manual two‐liner:
        mse  = mean_squared_error(preds.actual, preds.predicted)
        rmse = np.sqrt(mse)

        mae  = mean_absolute_error(preds.actual, preds.predicted)
        mape = (np.abs((preds.actual - preds.predicted) / preds.actual).mean() * 100)

        st.success(f"Done! 95% CI coverage: {coverage:.1%}")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse:.3f}")
        c2.metric("MAE",  f"{mae:.3f}")
        c3.metric("MAPE", f"{mape:.1f}%")

        # summary plots
        plot_matplotlib(df, preds)
        st.plotly_chart(plot_plotly(df, preds), use_container_width=True)

if __name__ == "__main__":
    main()
