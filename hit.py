import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class LinearRegressionApp:
    def __init__(self):
        self.initialize_session_state()
        self.create_layout()

    def initialize_session_state(self):
        # Initialize session state variables if they don't exist
        if 'data' not in st.session_state:
            st.session_state.data = pd.DataFrame({
                'X': [1, 2, 3, 4, 5],
                'Y': [2.1, 3.8, 4.9, 6.2, 7.1]
            })

    def calculate_regression(self):
        if len(st.session_state.data) < 2:
            return None

        x = st.session_state.data['X']
        y = st.session_state.data['Y']
        
        # Calculate regression parameters
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        # Calculate predicted values
        y_pred = slope * x + intercept
        
        # Calculate MSE and RMSE
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'mse': mse,
            'rmse': rmse,
            'y_pred': y_pred
        }

    def plot_regression(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = st.session_state.data['X']
        y = st.session_state.data['Y']
        
        # Plot scatter points
        ax.scatter(x, y, color='blue', alpha=0.5, label='Data Points')
        
        # Calculate and plot regression line if we have enough points
        if len(x) >= 2:
            reg_results = self.calculate_regression()
            if reg_results:
                x_range = np.array([min(x) - 0.5, max(x) + 0.5])
                y_pred = reg_results['slope'] * x_range + reg_results['intercept']
                ax.plot(x_range, y_pred, 'r-', label='Regression Line')
                
                # Plot residual lines
                for i in range(len(x)):
                    ax.plot([x.iloc[i], x.iloc[i]], 
                           [y.iloc[i], reg_results['y_pred'][i]], 
                           'g--', alpha=0.3)

        # Set plot properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Interactive Linear Regression')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        return fig

    def create_layout(self):
        st.title("Interactive Linear Regression Analysis")
        st.write("Analyze linear regression with interactive data manipulation")

        # Create sidebar for controls
        with st.sidebar:
            st.header("Controls")
            if st.button("Reset Data"):
                st.session_state.data = pd.DataFrame({
                    'X': [1, 2, 3, 4, 5],
                    'Y': [2.1, 3.8, 4.9, 6.2, 7.1]
                })
            if st.button("Clear Data"):
                st.session_state.data = pd.DataFrame(columns=['X', 'Y'])

        # Create main layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Regression Plot")
            fig = self.plot_regression()
            st.pyplot(fig)

        with col2:
            st.subheader("Statistics")
            reg_results = self.calculate_regression()
            if reg_results:
                st.write(f"Slope (β₁): {reg_results['slope']:.4f}")
                st.write(f"Y-intercept (β₀): {reg_results['intercept']:.4f}")
                st.write(f"R² Score: {reg_results['r_squared']:.4f}")
                st.write(f"MSE: {reg_results['mse']:.4f}")
                st.write(f"RMSE: {reg_results['rmse']:.4f}")
                st.write(f"Equation: y = {reg_results['slope']:.2f}x + {reg_results['intercept']:.2f}")

        # Data matrix section
        st.subheader("Data Points")
        
        # Create two columns for adding new points
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            new_x = st.number_input("X value", format="%.2f")
        with col2:
            new_y = st.number_input("Y value", format="%.2f")
        with col3:
            if st.button("Add Point"):
                new_data = pd.DataFrame({'X': [new_x], 'Y': [new_y]})
                st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)
                st.session_state.data = st.session_state.data.sort_values('X').reset_index(drop=True)

        # Display editable dataframe
        edited_df = st.data_editor(
            st.session_state.data,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True
        )
        
        # Update the session state with edited dataframe
        st.session_state.data = edited_df.sort_values('X').reset_index(drop=True)

def main():
    st.set_page_config(
        page_title="Linear Regression Analysis",
        layout="wide"
    )
    app = LinearRegressionApp()

if __name__ == "__main__":
    main()