import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Regression Metrics: R², RMSE, and Error Visualization")

# Split layout into two columns
col1, col2 = st.columns([2, 1])

with col1:
    
  # User controls
  n = st.slider("Sample Size", 1, 100, 10, 1)
  noise = st.slider("Noise Level", 0, 30, 3, 1)
  
  # Generate sample data
  np.random.seed(42)
  X = np.random.rand(n, 1) * 10
  
  # Create response variable with noise affecting only the variance
  model_component = 3 * X.squeeze() + 7
  y_noisy = model_component + np.random.randn(n) * noise
  
  data = pd.DataFrame({"X": X.squeeze(), "y": y_noisy})
  
  # Fit linear regression model (sklearn for predictions)
  model = LinearRegression()
  model.fit(X, y_noisy)
  y_pred = model.predict(X)
  
  # Fit regression model using statsmodels
  X_with_const = sm.add_constant(X)  # Add intercept term
  ols_model = sm.OLS(y_noisy, X_with_const).fit()
  summary_text = ols_model.summary().as_text()
  
  # Calculate metrics
  sse = np.sum((y_noisy - y_pred) ** 2)
  tse = np.sum((y_noisy - np.mean(y_noisy)) ** 2)
  r2 = r2_score(y_noisy, y_pred)
  rmse = np.sqrt(mean_squared_error(y_noisy, y_pred))
  
  # Define static y-limits based on max noise scenario
  y_min = (model_component.min() - 30).squeeze()
  y_max = (model_component.max() + 30).squeeze()
  
  # Plot data and regression line
  fig = px.scatter(x=X.squeeze(), y=y_noisy, labels={"x": "X", "y": "y"}, title="Regression Fit")
  fig.add_scatter(x=X.squeeze(), y=y_pred, mode='lines', name='Regression Line')
  fig.update_yaxes(range=[y_min, y_max])  # Set static y-axis limits
  
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

  
with col2:
  # Display formulas and corresponding metrics
  st.metric("Sum of Squared Errors (SSE)", f"{sse:.3f}")
  st.latex(r"SSE = \sum (y_i - \hat{y}_i)^2")
  
  st.metric("Total Sum of Squares (TSE)", f"{tse:.3f}")
  st.latex(r"TSE = \sum (y_i - \bar{y})^2")
  
  st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}")
  st.latex(r"RMSE = \sqrt{\frac{SSE}{n}}")
  
  st.metric("Coefficient of Determination (R²)", f"{r2:.3f}")
  st.latex(r"R^2 = 1 - \frac{SSE}{TSE}")
  
  # Display regression summary from statsmodels
  st.text(summary_text)
