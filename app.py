# app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Set up the Streamlit App Title and Sidebar ---
st.set_page_config(page_title="Linear Regression Visualizer", layout="centered")
st.title("📊 Interactive Linear Regression Visualizer")
st.markdown("Adjust parameters on the sidebar to observe changes in data generation, regression line fitting, and outlier identification.")

# 1. Sidebar Inputs
st.sidebar.header("⚙️ Data Generation Parameters")

# Number of data points (n)
n_points = st.sidebar.slider("Number of Data Points (n)", 100, 1000, 500)

# True coefficient (a)
true_a = st.sidebar.slider("True Coefficient (a)", -10.0, 10.0, 2.5, 0.1)

# Variance for noise (var)
variance = st.sidebar.slider("Noise Variance (var)", 0, 1000, 200, 10)

# Fixed true intercept (b)
true_b = 5.0

st.sidebar.markdown(f"---")
st.sidebar.markdown(f"**True Model:** $y = {true_a:.2f}x + {true_b:.2f} + noise$")

# --- Function for Data Generation (Cached for efficiency) ---
@st.cache_data
def generate_data(n, a, b, var):
    """Generates synthetic data: y = ax + b + N(0, var)"""
    # 1. Generate x values
    X = np.random.uniform(low=0, high=100, size=(n, 1))

    # 2. Generate noise (np.sqrt(var) is the standard deviation)
    noise = np.random.normal(loc=0, scale=np.sqrt(var), size=(n, 1))

    # 3. Calculate y
    y = a * X + b + noise

    # Return DataFrame and original numpy arrays
    df = pd.DataFrame(X, columns=['X'])
    df['y'] = y
    return df, X, y

# --- Generate Data ---
data_df, X, y = generate_data(n_points, true_a, true_b, variance)

# --- Linear Regression Model Training (P2.1) ---
model = LinearRegression()
model.fit(X, y)

# Get the calculated coefficients
calc_a = model.coef_[0][0]
calc_b = model.intercept_[0]

# Make predictions
y_pred = model.predict(X)

# Add predictions to the DataFrame
data_df['y_pred'] = y_pred

# --- Outlier Identification (P2.2) ---
data_df['Residual'] = data_df['y'] - data_df['y_pred']
data_df['Abs_Residual'] = np.abs(data_df['Residual'])

# Identify the Top 5 Outliers
outliers_df = data_df.nlargest(5, 'Abs_Residual')


# --- Display Regression Results (P3.3) ---
st.subheader("📋 Regression Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Fitted Coefficient $\\hat{a}$",
        value=f"{calc_a:.4f}",
        delta=f"Diff: {(calc_a - true_a):.4f}" # Show difference from true a
    )

with col2:
    st.metric(
        label="R-squared ($\mathbf{R^2}$) Score",
        value=f"{model.score(X, y):.4f}",
        help="Proportion of the variance in the dependent variable that is predictable from the independent variable(s). Closer to 1 is better."
    )
    
st.markdown(f"**Fitted Model:** $\\hat{{y}} = {calc_a:.3f}x + {calc_b:.3f}$")
st.markdown("---")

# --- Visualization (P2.3 & P2.4) ---
st.subheader("📈 Data Points and Regression Line")

fig, ax = plt.subplots(figsize=(10, 6))

# 1. Plot all data points
ax.scatter(data_df['X'], data_df['y'], label='Data Points', s=20, alpha=0.6, color='blue')

# 2. Draw the calculated linear regression line in RED
ax.plot(
    data_df['X'],
    data_df['y_pred'],
    color='red',
    linewidth=3,
    label=f'Regression Line $\\hat{{y}}$'
)

# 3. Highlight and label the outliers
ax.scatter(
    outliers_df['X'],
    outliers_df['y'],
    color='orange',
    s=150,
    edgecolors='black',
    marker='o',
    label='Outliers (Top 5)',
    zorder=5
)

# Add labels to the outliers
for i, row in outliers_df.iterrows():
    ax.annotate(
        f'Outlier {i}', # Use the original index as a label
        (row['X'], row['y']),
        textcoords="offset points",
        xytext=(5, -10),
        ha='left',
        fontsize=9,
        color='black',
        fontweight='bold'
    )

# Final Plot Polish
ax.set_title("Linear Regression Visualization with Outlier Identification", fontsize=14)
ax.set_xlabel("X Value")
ax.set_ylabel("Y Value")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.7)

# 4. Display the plot in Streamlit
st.pyplot(fig)

st.markdown("---")
st.subheader("📌 Top 5 Outliers List")
st.dataframe(outliers_df[['X', 'y', 'Residual', 'Abs_Residual']].sort_values(by='Abs_Residual', ascending=False).style.format(precision=3))