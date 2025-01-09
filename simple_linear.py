import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title
st.title("Simple Linear Regression Example")
st.write("This app demonstrates linear regression using the California Housing Prices dataset.")

# Load Dataset
#@st.cache
def load_data():
    return pd.read_csv("housing.csv")

data = load_data()

# Display dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Sidebar for selecting features
st.sidebar.header("Select Features")
x_feature = st.sidebar.selectbox("Select the feature for X (independent variable)", data.columns)
y_feature = st.sidebar.selectbox("Select the feature for Y (dependent variable)", data.columns)

# Prepare data for regression
X = data[[x_feature]]
Y = data[y_feature]

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make predictions
Y_pred = model.predict(X_test)

# Model performance
mse = mean_squared_error(Y_test, Y_pred)
st.subheader("Model Performance")
st.write(f"Mean Squared Error (MSE): {mse:.2f}")

# Visualization
st.subheader("Visualization")
fig, ax = plt.subplots()
ax.scatter(X_test, Y_test, color="blue", label="Actual")
ax.plot(X_test, Y_pred, color="red", label="Predicted")
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_title("Linear Regression")
ax.legend()
st.pyplot(fig)

# Prediction
st.sidebar.header("Prediction")
user_value = st.sidebar.number_input(f"Enter a value for {x_feature}", min_value=float(X.min()), max_value=float(X.max()), value=float(X.mean()))
predicted_value = model.predict([[user_value]])[0]
st.sidebar.write(f"Predicted {y_feature}: {predicted_value:.2f}")
