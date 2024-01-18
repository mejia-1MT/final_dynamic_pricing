import tkinter as tk
from tkinter import ttk
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Scrollbar, messagebox

def display_prices_graph(d_prices, s_prices, d_revenues, s_revenues, demands, predicted_price_31, predicted_demand_31):
    days = list(range(1, 31))
    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Performance Analysis")

    # Add a blank space on the left for text
    text_frame = tk.Frame(root, width=400, height=400, bg='blue')  # Left space for text
    text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=0.8)

    # Create subplots for prices and revenues with padding
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

   # Plot dynamic prices (upper left)
    axes[0, 0].plot(days, d_prices, label='Dynamic Price', marker='o')
    axes[0, 0].set_title('Dynamic Price')
    axes[0, 0].set_xlabel('Day')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].margins(0.1)  # Add padding
    axes[0, 0].xaxis.set_major_locator(plt.MultipleLocator(5))  # Add x-axis grid every 5 days
    axes[0, 0].grid(True)  # Add grid lines

    # Plot static prices (upper right)
    axes[0, 1].plot(days, s_prices, label='Static Price', marker='o')
    axes[0, 1].set_title('Static Price')
    axes[0, 1].set_xlabel('Day')
    axes[0, 1].set_ylabel('Price')
    axes[0, 1].margins(0.1)  # Add padding
    axes[0, 1].xaxis.set_major_locator(plt.MultipleLocator(5))  # Add x-axis grid every 5 days
    axes[0, 1].grid(True)  # Add grid lines

    # Plot dynamic revenues (lower left)
    axes[1, 0].plot(days, d_revenues, label='Dynamic Revenue', marker='o')
    axes[1, 0].set_title('Dynamic Revenue')
    axes[1, 0].set_xlabel('Day')
    axes[1, 0].set_ylabel('Revenue')
    axes[1, 0].margins(0.1)  # Add padding
    axes[1, 0].xaxis.set_major_locator(plt.MultipleLocator(5))  # Add x-axis grid every 5 days
    axes[1, 0].grid(True)  # Add grid lines

    # Plot static revenues (lower right)
    axes[1, 1].plot(days, s_revenues, label='Static Revenue', marker='o')
    axes[1, 1].set_title('Static Revenue')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Revenue')
    axes[1, 1].margins(0.1)  # Add padding
    axes[1, 1].xaxis.set_major_locator(plt.MultipleLocator(5))  # Add x-axis grid every 5 days
    axes[1, 1].grid(True)  # Add grid lines

    # Add legend
    for ax in axes.flat:
        ax.legend()

    # Create a canvas for the plots
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    # Display dynamic and static pricing breakdown on the right text_frame
    dynamic_text = "\nDynamic Pricing Breakdown\n\n"
    for day, (d_price, d_revenue, demand) in enumerate(zip(d_prices, d_revenues, demands), start=1):
        dynamic_text += f"Day {day} Price: {d_price:.2f} Demand: {demand:.2f} Revenue: {d_revenue:.2f}\n"

    static_text = "\nStatic Pricing Breakdown\n\n"
    for day, (s_price, s_revenue, demand) in enumerate(zip(s_prices, s_revenues, demands), start=1):
        static_text += f"Day {day} Price: {s_price:.2f} Demand: {demand:.2f} Revenue: {s_revenue:.2f}\n"

    # Append the revenue print statements to the static text
    static_text += f"\n\nTotal Revenue for Dynamic Pricing: {sum(d_revenues):.2f}\n"
    static_text += f"Total Revenue for Static Pricing: {sum(s_revenues):.2f}\n\n\n"

    # Create a Text widget and Scrollbar
    text_widget = tk.Text(text_frame, wrap="none", font=("Arial", 12), height=20, width=45)  # Change the font here
    scrollbar = tk.Scrollbar(text_frame, command=text_widget.yview)
    text_widget.config(yscrollcommand=scrollbar.set)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Insert the text into the Text widget
    text_widget.insert(tk.END, dynamic_text + "\n" + static_text)

     # Function to show the dialog box
    def show_prediction():
        messagebox.showinfo("Prediction for Day 31", f"Predicted Demand: {predicted_demand_31:.2f}\nPredicted Price: {predicted_price_31:.2f}")
        # Close the "Product's cost" window
        root.destroy()
    # Add a button to the center on top of all elements
    button = ttk.Button(root, text="Predict day 31!", command=show_prediction)
    button.place(relx=.14, rely=.95, anchor=tk.N)

    # Start the Tkinter event loop
    root.mainloop()