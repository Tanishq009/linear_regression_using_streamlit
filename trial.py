import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    st.title('California Housing Price Prediction')
    
    # Load data
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    
    # Let user select feature for prediction
    feature = st.selectbox('Select feature for prediction:', housing.feature_names)
    
    # Prepare data
    X = df[[feature]]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Create plot
    fig, ax = plt.subplots()
    ax.scatter(X_test, y_test, alpha=0.5)
    ax.plot(X_test, model.predict(X_test), color='red')
    ax.set_xlabel(feature)
    ax.set_ylabel('Price')
    
    # Show results
    st.pyplot(fig)
    st.write(f'R² Score: {model.score(X_test, y_test):.3f}')
    
    # Interactive prediction
    user_input = st.number_input(f'Enter {feature} value:', value=float(X[feature].mean()))
    prediction = model.predict([[user_input]])[0]
    st.write(f'Predicted Price: ${prediction:.2f}k')

if __name__ == '__main__':
    main()