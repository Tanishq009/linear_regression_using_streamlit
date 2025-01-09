import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Regression Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS with improved color scheme and consistent styling
st.markdown("""
    <style>
        /* Main background and text colors */
        .stApp {
            background-color: #f8f9fa;
            color: #212529;
        }
        
        /* Container styling */
        .main {
            padding: 2rem;
        }
        
        /* Card styling */
        .st-emotion-cache-1v0mbdj {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        /* Metrics styling */
        div[data-testid="stMetricValue"] {
            color: #2c3e50;
            font-weight: 600;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border-radius: 5px;
        }
        
        /* DataFrame styling */
        .dataframe {
            font-size: 0.9rem;
            color: #2c3e50;
        }
        
        /* Success message styling */
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #c3e6cb;
        }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['Price'] = housing.target
    return df

def main():
    # Title and introduction with improved styling
    st.title('🏠 California Housing Price Prediction')
    st.markdown("""
    <div style='background-color: #e9ecef; padding: 1rem; border-radius: 5px; margin-bottom: 2rem;'>
        This interactive dashboard performs regression analysis on the California Housing dataset.
        Explore the data, visualize relationships, and see the prediction results.
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar with improved styling
    st.sidebar.markdown("""
        <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 5px;'>
            <h2 style='color: #2c3e50; font-size: 1.5rem;'>Analysis Controls</h2>
        </div>
    """, unsafe_allow_html=True)
    
    feature = st.sidebar.selectbox(
        'Select Feature for Analysis',
        df.drop('Price', axis=1).columns
    )
    
    test_size = st.sidebar.slider(
        'Test Set Size',
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.1
    )
    
    # Main content with improved layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📊 Dataset Overview')
        st.dataframe(df.head(), use_container_width=True)
        
        st.subheader('📈 Feature Distribution')
        fig_dist = px.histogram(
            df, 
            x=feature, 
            nbins=30,
            title=f'Distribution of {feature}',
            template='simple_white',
            color_discrete_sequence=['#2c3e50']
        )
        fig_dist.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_color='#2c3e50',
            font_color='#2c3e50'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader('📉 Statistical Summary')
        st.dataframe(df.describe(), use_container_width=True)
        
        st.subheader('🔍 Correlation with Price')
        fig_scatter = px.scatter(
            df, 
            x=feature, 
            y='Price',
            title=f'Price vs {feature}',
            template='simple_white',
            color_discrete_sequence=['#2c3e50']
        )
        fig_scatter.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            title_font_color='#2c3e50',
            font_color='#2c3e50'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Regression Analysis with improved styling
    st.markdown("""
        <h2 style='color: #2c3e50; margin-top: 2rem;'>🎯 Regression Analysis</h2>
    """, unsafe_allow_html=True)
    
    # Prepare data for regression
    X = df[[feature]]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Model metrics with improved styling
    metrics_container = st.container()
    col3, col4, col5 = metrics_container.columns(3)
    
    metric_style = """
        <div style='background-color: white; padding: 1rem; border-radius: 5px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;'>
            <h4 style='color: #2c3e50; margin-bottom: 0.5rem;'>{}</h4>
            <p style='color: #2c3e50; font-size: 1.5rem; font-weight: bold;'>{}</p>
        </div>
    """
    
    with col3:
        st.markdown(metric_style.format("R² Score", f"{r2_score(y_test, y_pred):.3f}"), 
                   unsafe_allow_html=True)
    
    with col4:
        st.markdown(metric_style.format("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.3f}"), 
                   unsafe_allow_html=True)
    
    with col5:
        st.markdown(metric_style.format("Coefficient", f"{model.coef_[0]:.3f}"), 
                   unsafe_allow_html=True)
    
    # Regression plot with improved styling
    fig_regression = go.Figure()
    
    # Add scatter plot of actual data
    fig_regression.add_trace(
        go.Scatter(
            x=X_test[feature],
            y=y_test,
            mode='markers',
            name='Actual',
            marker=dict(color='#2c3e50', size=8, opacity=0.6)
        )
    )
    
    # Add regression line
    X_line = np.linspace(X_test[feature].min(), X_test[feature].max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    
    fig_regression.add_trace(
        go.Scatter(
            x=X_line.flatten(),
            y=y_line,
            mode='lines',
            name='Regression Line',
            line=dict(color='#e74c3c', width=2)
        )
    )
    
    fig_regression.update_layout(
        title='Regression Analysis Results',
        xaxis_title=feature,
        yaxis_title='Price',
        plot_bgcolor='white',
        paper_bgcolor='white',
        title_font_color='#2c3e50',
        font_color='#2c3e50',
        hovermode='closest'
    )
    
    st.plotly_chart(fig_regression, use_container_width=True)
    
    # Prediction tool with improved styling
    st.markdown("""
        <h3 style='color: #2c3e50; margin-top: 2rem;'>🎯 Prediction Tool</h3>
    """, unsafe_allow_html=True)
    
    user_input = st.slider(
        f'Select {feature} value',
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )
    
    prediction = model.predict([[user_input]])[0]
    st.markdown(f"""
        <div style='background-color: #d4edda; color: #155724; padding: 1rem; 
                    border-radius: 5px; border: 1px solid #c3e6cb; text-align: center;'>
            <h4 style='margin-bottom: 0.5rem;'>Predicted Price</h4>
            <p style='font-size: 1.5rem; font-weight: bold;'>${prediction:,.2f}</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()