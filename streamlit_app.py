import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide")

st.title("Regression Metrics: R², RMSE, and Error Visualization")

# Split layout into two columns
col1, col2 = st.columns([1, 1])

with col1:
    
  # User controls
  b0 = st.slider("$\Beta_0$", -5, 5, 3, 1)
  b1 = st.slider("$\Beta_1$", -10, 10, 3, 1)
  n = st.slider("Sample Size", 1, 100, 10, 1)
  noise = st.slider("Noise Level", 0, 30, 3, 1)
  
  # Generate sample data
  np.random.seed(42)
  X = np.random.rand(n, 1) * 20 - 10
  
  # Create response variable with noise affecting only the variance
  model_component = b0 + b1 * X.squeeze()
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
  SSE = np.sum((y_noisy - y_pred) ** 2)
  SST = np.sum((y_noisy - np.mean(y_noisy)) ** 2)
  MSE = SSE / n
  variance_y = SST / n
  r2 = 1 - (SSE / SST)
  rmse = np.sqrt(MSE)
  
  # Define static y-limits based on max noise scenario
  y_min = (model_component.min() - 30).squeeze()
  y_max = (model_component.max() + 30).squeeze()

  # Plot data
  fig = px.scatter(x=X.squeeze(), y=y_noisy, labels={"x": "X", "y": "y"}, title="Regression Fit")
  
  # Add selected error visualizations
  show_sse = st.checkbox("Show SSE (Sum of Squares Error)")
  show_sst = st.checkbox("Show SST (Total Sum of Squares)")

  if show_sst:
      for i in range(n):
          fig.add_trace(go.Scatter(x=[X.squeeze()[i], X.squeeze()[i]],
                                   y=[y_noisy[i], np.mean(y_noisy)],
                                   mode='lines', line=dict(color='blue'),
                                   showlegend=(i == 0), name='SST'))
      fig.add_scatter(x=X.squeeze(), y=[np.mean(y_noisy)]*n, mode='lines', name='Mean Y', line=dict(dash='dash', color='green'))
      
  if show_sse:
      for i in range(n):
          fig.add_trace(go.Scatter(x=[X.squeeze()[i], X.squeeze()[i]],
                                   y=[y_noisy[i], y_pred[i]],
                                   mode='lines', line=dict(color='red', dash='dot'),
                                   showlegend=(i == 0), name='SSE'))

  # Plot data and regression line
  fig.add_scatter(x=X.squeeze(), y=y_pred, mode='lines', name='Regression Line', line=dict(color="orange"))
  fig.update_yaxes(range=[y_min, y_max])  # Set static y-axis limits

  
  st.plotly_chart(fig)

  
with col2:
  # Display formulas and corresponding metrics
  st.metric("Sum of Squares Error (SSE)", f"{SSE:.3f}")
  st.latex(r"SSE = \sum (y_i - \hat{y}_i)^2")

  st.metric("Mean Squared Error (MSE)", f"{MSE:.3f}")
  st.latex(r"MSE = \frac{SSE}{n}")
  
  st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.3f}")
  st.latex(r"RMSE = \sqrt{MSE}")

  st.metric("Variance of Y", f"{variance_y:.3f}")
  st.latex(r"Var(Y) = \frac{SST}{n}")

  st.metric("Total Sum of Squares (SST)", f"{SST:.3f}")
  st.latex(r"SST = \sum (y_i - \bar{y})^2")
  
  st.metric("Coefficient of Determination (R²)", f"{r2:.3f}")
  st.latex(r"R^2 = 1 - \frac{SSE}{SST} = 1 - \frac{MSE}{Var(Y)}")
  
  # Display regression summary from statsmodels
  st.text(summary_text)
