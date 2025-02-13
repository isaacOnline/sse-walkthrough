import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Regression Metrics: R², RMSE, and Error Visualization")

# Generate sample data
np.random.seed(42)
n = 35
X = np.random.rand(n, 1) * 10
y_true = 3 * X.squeeze() + 7 + np.random.randn(n) * 3

data = pd.DataFrame({"X": X.squeeze(), "y": y_true})

# User controls for noise level
noise = st.slider("Noise Level", 0.0, 5.0, 3.0, 0.1)
y_noisy = 3 * X.squeeze() + 7 + np.random.randn(n) * noise

# Fit linear regression model
model = LinearRegression()
model.fit(X, y_noisy)
y_pred = model.predict(X)

# Calculate metrics
sse = np.sum((y_noisy - y_pred) ** 2)
tse = np.sum((y_noisy - np.mean(y_noisy)) ** 2)
r2 = r2_score(y_noisy, y_pred)
rmse = np.sqrt(mean_squared_error(y_noisy, y_pred))

# Plot data and regression line
fig = px.scatter(x=X.squeeze(), y=y_noisy, labels={"x": "X", "y": "y"}, title="Regression Fit")
fig.add_scatter(x=X.squeeze(), y=y_pred, mode='lines', name='Regression Line')

# Add selected error visualizations
show_sse = st.checkbox("Show Sum of Squared Errors (SSE)")
show_tse = st.checkbox("Show Total Squared Error (TSE)")

if show_sse:
    for i in range(n):
        fig.add_trace(go.Scatter(x=[X.squeeze()[i], X.squeeze()[i]],
                                 y=[y_noisy[i], y_pred[i]],
                                 mode='lines', line=dict(color='red'),
                                 showlegend=(i == 0), name='SSE (Residual)'))
if show_tse:
    for i in range(n):
        fig.add_trace(go.Scatter(x=[X.squeeze()[i], X.squeeze()[i]],
                                 y=[y_noisy[i], np.mean(y_noisy)],
                                 mode='lines', line=dict(color='blue'),
                                 showlegend=(i == 0), name='TSE'))

st.plotly_chart(fig)

# Display formulas and corresponding metrics
st.latex(r"SSE = \sum (y_i - \hat{y}_i)^2")
st.metric("Sum of Squared Errors (SSE)", f"{sse:.3f}")

st.latex(r"TSE = \sum (y_i - \bar{y})^2")
st.metric("Total Sum of Squares (TSE)", f"{tse:.3f}")

st.latex(r"RMSE = \sqrt{\frac{SSE}{n}}")
st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}")

st.latex(r"R^2 = 1 - \frac{SSE}{TSE}")
st.metric("Coefficient of Determination (R²)", f"{r2:.3f}")
