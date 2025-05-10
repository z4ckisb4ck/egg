import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.express as px
from sklearn.model_selection import train_test_split

# Static historical data
BLS_DATA = np.array([
    [2017, 1, 1.47], [2017, 2, 1.48], [2017, 3, 1.49], [2017, 4, 1.50], [2017, 5, 1.51], [2017, 6, 1.52],
    [2017, 7, 1.53], [2017, 8, 1.54], [2017, 9, 1.55], [2017, 10, 1.56], [2017, 11, 1.57], [2017, 12, 1.58],
    [2018, 1, 1.74], [2018, 2, 1.75], [2018, 3, 1.76], [2018, 4, 1.77], [2018, 5, 1.78], [2018, 6, 1.79],
    [2018, 7, 1.80], [2018, 8, 1.81], [2018, 9, 1.82], [2018, 10, 1.83], [2018, 11, 1.84], [2018, 12, 1.85],
    [2019, 1, 1.40], [2019, 2, 1.41], [2019, 3, 1.42], [2019, 4, 1.43], [2019, 5, 1.44], [2019, 6, 1.45],
    [2019, 7, 1.46], [2019, 8, 1.47], [2019, 9, 1.48], [2019, 10, 1.49], [2019, 11, 1.50], [2019, 12, 1.51],
    [2020, 1, 1.461], [2020, 2, 1.449], [2020, 3, 1.525], [2020, 4, 2.019], [2020, 5, 1.640], [2020, 6, 1.554],
    [2020, 7, 1.401], [2020, 8, 1.328], [2020, 9, 1.353], [2020, 10, 1.408], [2020, 11, 1.450], [2020, 12, 1.481],
    [2021, 1, 1.466], [2021, 2, 1.597], [2021, 3, 1.625], [2021, 4, 1.620], [2021, 5, 1.625], [2021, 6, 1.642],
    [2021, 7, 1.642], [2021, 8, 1.709], [2021, 9, 1.835], [2021, 10, 1.821], [2021, 11, 1.718], [2021, 12, 1.788],
])

# Convert data to DataFrame
def load_data():
    df = pd.DataFrame(BLS_DATA, columns=["Year", "Month", "Price"])
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
    return df

# Train a simple XGBoost model
def train_model(df):
    df["Month"] = df["Month"].astype(int)
    X = df[["Year", "Month"]]
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Streamlit app
def main():
    st.title("Egg Price Prediction")
    st.write("This app predicts egg prices based on historical data.")

    # Load and display data
    df = load_data()
    st.write("### Historical Data", df)

    # Plot historical data
    fig = px.line(df, x="Date", y="Price", title="Historical Egg Prices")
    st.plotly_chart(fig)

    # Train model
    model, X_test, y_test = train_model(df)
    st.write("### Model Trained Successfully")

    # Predict and display results
    predictions = model.predict(X_test)
    results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
    st.write("### Predictions vs Actual", results)

if __name__ == "__main__":
    main()