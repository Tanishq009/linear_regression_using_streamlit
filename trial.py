import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from scipy import stats

class InteractiveLinearRegression:
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Interactive Linear Regression")
        self.root.geometry("1000x800")

        # Initialize data
        self.x_data = np.array([1, 2, 3, 4, 5])
        self.y_data = np.array([2.1, 3.8, 4.9, 6.2, 7.1])

        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.create_layout()
        self.update_plot()

        # Bind click event
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def create_layout(self):
        # Create main frames
        self.plot_frame = ttk.Frame(self.root)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.stats_frame = ttk.Frame(self.root)
        self.stats_frame.pack(fill=tk.X, padx=10, pady=5)

        self.controls_frame = ttk.Frame(self.root)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=5)

        # Add matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create statistics labels
        self.slope_var = tk.StringVar()
        self.intercept_var = tk.StringVar()
        self.r2_var = tk.StringVar()
        self.equation_var = tk.StringVar()
        self.mse_var = tk.StringVar()  # New variable for MSE
        self.rmse_var = tk.StringVar()  # New variable for RMSE

        # Statistics display
        stats_labels = ttk.LabelFrame(self.stats_frame, text="Regression Statistics")
        stats_labels.pack(fill=tk.X, pady=5)

        ttk.Label(stats_labels, textvariable=self.slope_var).pack(anchor='w', padx=5)
        ttk.Label(stats_labels, textvariable=self.intercept_var).pack(anchor='w', padx=5)
        ttk.Label(stats_labels, textvariable=self.r2_var).pack(anchor='w', padx=5)
        ttk.Label(stats_labels, textvariable=self.equation_var).pack(anchor='w', padx=5)
        ttk.Label(stats_labels, textvariable=self.mse_var).pack(anchor='w', padx=5)
        ttk.Label(stats_labels, textvariable=self.rmse_var).pack(anchor='w', padx=5)

        # Control buttons
        ttk.Button(self.controls_frame, text="Reset Data", 
                  command=self.reset_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.controls_frame, text="Clear All", 
                  command=self.clear_data).pack(side=tk.LEFT, padx=5)

    def calculate_mse(self, slope, intercept):
        # Calculate predicted values
        y_pred = slope * self.x_data + intercept
        
        # Calculate MSE
        mse = np.mean((self.y_data - y_pred) ** 2)
        
        # Calculate RMSE
        rmse = np.sqrt(mse)
        
        # Update MSE display
        self.mse_var.set(f"Mean Squared Error (MSE): {mse:.4f}")
        self.rmse_var.set(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        return mse, rmse

    def calculate_regression(self):
        if len(self.x_data) < 2:
            return None
        
        # Calculate regression parameters
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.x_data, self.y_data)
        r_squared = r_value ** 2

        # Calculate MSE
        mse, rmse = self.calculate_mse(slope, intercept)

        # Update statistics display
        self.slope_var.set(f"Slope (β₁): {slope:.4f}")
        self.intercept_var.set(f"Y-intercept (β₀): {intercept:.4f}")
        self.r2_var.set(f"R² (Coefficient of Determination): {r_squared:.4f}")
        self.equation_var.set(f"Equation: y = {slope:.2f}x + {intercept:.2f}")

        return slope, intercept, r_squared

    def update_plot(self):
        self.ax.clear()
        
        # Plot scatter points
        self.ax.scatter(self.x_data, self.y_data, color='blue', alpha=0.5, label='Data Points')
        
        # Calculate and plot regression line if we have enough points
        if len(self.x_data) >= 2:
            reg_results = self.calculate_regression()
            if reg_results:
                slope, intercept, _ = reg_results
                x_range = np.array([min(self.x_data) - 0.5, max(self.x_data) + 0.5])
                y_pred = slope * x_range + intercept
                self.ax.plot(x_range, y_pred, 'r-', label='Regression Line')
                
                # Plot residual lines
                y_pred_points = slope * self.x_data + intercept
                for x, y, y_p in zip(self.x_data, self.y_data, y_pred_points):
                    self.ax.plot([x, x], [y, y_p], 'g--', alpha=0.3)

        # Set plot properties
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Interactive Linear Regression\nClick to Add Points')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.legend()

        # Update plot limits with some padding
        if len(self.x_data) > 0:
            x_margin = (max(self.x_data) - min(self.x_data)) * 0.1 if len(self.x_data) > 1 else 1
            y_margin = (max(self.y_data) - min(self.y_data)) * 0.1 if len(self.y_data) > 1 else 1
            self.ax.set_xlim(min(self.x_data) - x_margin, max(self.x_data) + x_margin)
            self.ax.set_ylim(min(self.y_data) - y_margin, max(self.y_data) + y_margin)
        else:
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, 10)

        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        # Add new point
        x, y = event.xdata, event.ydata
        
        # Check if we're updating an existing x-coordinate
        x_index = np.where(np.isclose(self.x_data, x, atol=0.1))[0]
        if len(x_index) > 0:
            # Update existing point
            self.y_data[x_index[0]] = y
        else:
            # Add new point
            self.x_data = np.append(self.x_data, x)
            self.y_data = np.append(self.y_data, y)
            
            # Sort data by x-values
            sort_indices = np.argsort(self.x_data)
            self.x_data = self.x_data[sort_indices]
            self.y_data = self.y_data[sort_indices]

        self.update_plot()

    def reset_data(self):
        # Reset to initial data
        self.x_data = np.array([1, 2, 3, 4, 5])
        self.y_data = np.array([2.1, 3.8, 4.9, 6.2, 7.1])
        self.update_plot()

    def clear_data(self):
        # Clear all data
        self.x_data = np.array([])
        self.y_data = np.array([])
        self.update_plot()
        
        # Clear statistics
        self.slope_var.set("Slope (β₁): -")
        self.intercept_var.set("Y-intercept (β₀): -")
        self.r2_var.set("R² (Coefficient of Determination): -")
        self.equation_var.set("Equation: -")
        self.mse_var.set("Mean Squared Error (MSE): -")
        self.rmse_var.set("Root Mean Squared Error (RMSE): -")

    def run(self):
        self.root.mainloop()

# Create and run the application
if __name__ == "__main__":
    app = InteractiveLinearRegression()
    app.run()