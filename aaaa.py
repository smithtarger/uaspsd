import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
import numpy as np
import streamlit as st

# Sidebar controls
st.sidebar.header("Configuration")
test_size = st.sidebar.slider("Test Size", 0.0, 1.0, 0.2, 0.1)
random_state = st.sidebar.number_input("Random State", value=42)

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset")# Display the dataset
    st.dataframe(df)

    df = df.dropna()

    # Assuming you have defined your DataFrame `df` and selected the features and target variable
    X = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    y = df['Close']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    st.write("Splitting the dataset into training and test sets...")
    st.write("Training set size:", len(X_train))
    st.write("Test set size:", len(X_test))

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.write("Scaling the features using StandardScaler...")
    st.write("Scaled features:")
    st.write(X_train_scaled)

    # Create an empty DataFrame to store the MAPE results
    mape_df = pd.DataFrame(columns=['Model', 'MAPE'])

    # Create the composite kernel
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

    # Create the Gaussian process regressor with the composite kernel
    regressor_gpr = GaussianProcessRegressor(kernel=kernel, random_state=42)
    regressor_gpr.fit(X_train_scaled, y_train)
    y_pred_gpr = regressor_gpr.predict(X_test_scaled)
    mape_gpr = mean_absolute_percentage_error(y_test, y_pred_gpr)
    mape_df = mape_df.append({'Model': 'Gaussian Process Regressor', 'MAPE': mape_gpr}, ignore_index=True)

    # Create the KNeighborsRegressor model
    regressor_knn = KNeighborsRegressor(n_neighbors=5)
    regressor_knn.fit(X_train_scaled, y_train)
    y_pred_knn = regressor_knn.predict(X_test_scaled)
    mape_knn = mean_absolute_percentage_error(y_test, y_pred_knn)
    mape_df = mape_df.append({'Model': 'KNeighborsRegressor', 'MAPE': mape_knn}, ignore_index=True)

    # Create the RandomForestRegressor model
    regressor_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor_rf.fit(X_train_scaled, y_train)
    y_pred_rf = regressor_rf.predict(X_test_scaled)
    mape_rf = mean_absolute_percentage_error(y_test, y_pred_rf)
    mape_df = mape_df.append({'Model': 'RandomForestRegressor', 'MAPE': mape_rf}, ignore_index=True)

    # Initialize the LinearSVR and Ridge models
    estimators = [
        ('svr', LinearSVR(random_state=42)),
        ('ridge', Ridge(random_state=42))
    ]

    # Initialize the StackingRegressor
    regressor_stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    regressor_stack.fit(X_train_scaled, y_train)
    y_pred_stack = regressor_stack.predict(X_test_scaled)
    mape_stack = mean_absolute_percentage_error(y_test, y_pred_stack)
    mape_df = mape_df.append({'Model': 'StackingRegressor', 'MAPE': mape_stack}, ignore_index=True)

    # Create the DecisionTreeRegressor model
    regressor_dt = DecisionTreeRegressor(random_state=42)
    regressor_dt.fit(X_train_scaled, y_train)
    y_pred_dt = regressor_dt.predict(X_test_scaled)
    mape_dt = mean_absolute_percentage_error(y_test, y_pred_dt)
    mape_df = mape_df.append({'Model': 'DecisionTreeRegressor', 'MAPE': mape_dt}, ignore_index=True)

    # Create the LinearRegression model
    regressor_lr = LinearRegression()
    regressor_lr.fit(X_train_scaled, y_train)
    y_pred_lr = regressor_lr.predict(X_test_scaled)
    mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)
    mape_df = mape_df.append({'Model': 'LinearRegression', 'MAPE': mape_lr}, ignore_index=True)

    # Initialize the DummyRegressor
    regressor_dummy = DummyRegressor(strategy='mean')
    regressor_dummy.fit(X_train_scaled, y_train)
    y_pred_dummy = regressor_dummy.predict(X_test_scaled)
    mape_dummy = mean_absolute_percentage_error(y_test, y_pred_dummy)
    mape_df = mape_df.append({'Model': 'DummyRegressor', 'MAPE': mape_dummy}, ignore_index=True)

    
    # Display the MAPE results in a DataFrame
    st.write("Mean Absolute Percentage Error (MAPE) for Each Model:")
    st.dataframe(mape_df)
    
 

