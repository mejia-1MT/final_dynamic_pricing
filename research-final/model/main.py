from functools import partial
from tkinter import messagebox
import torch
import os
import pandas as pd
from environment import PricingEnvironment
from dqn_agent import DQNAgent
from simulate_episode import simulate
from train import training
from data_prep import load_dataset, preprocess_data
import tkinter as tk
from tkinter import simpledialog
from tkinter import ttk
from sklearn.preprocessing import MinMaxScaler
from calculate_demand import predict_daily_demand, calculate_average_customers

def train_model(baseline_price):
    try:
        # Perform training
        env = PricingEnvironment(scaled_data, daily_demand, baseline_price)
        agent = DQNAgent(clipped_features, 50, baseline_price)
        training(env, agent)
        messagebox.showinfo("Training Complete", "Training completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error during training: {str(e)}")

def simulate_model(baseline_price):
    try:
        # Perform simulation
        env = PricingEnvironment(scaled_data, daily_demand, baseline_price)
        agent = DQNAgent(clipped_features, 50, baseline_price)
        simulate(env, agent)
        messagebox.showinfo("Simulation Complete", "Simulation completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Error during simulation: {str(e)}")

def get_baseline_price_and_run(selected_method):
    for widget in root.winfo_children():
        widget.destroy()

    baseline_label = ttk.Label(root, text="Enter the Baseline Price:")
    baseline_label.grid(row=0, column=0, padx=80, pady=10)

    baseline_entry = ttk.Entry(root)
    baseline_entry.grid(row=1, column=0, padx=80, pady=10)

    submit_button = ttk.Button(root, text="Submit", command=lambda: on_submit(selected_method, baseline_entry.get()))
    submit_button.grid(row=2, column=0, padx=80, pady=10)

    # Change the title after setting up the new elements
    root.title("Product's cost")

def on_submit(selected_method, baseline_price):
    try:
        # Close the window after completing the operation
        root.destroy()

        baseline_price = float(baseline_price)
        if selected_method == "Train":
            train_model(baseline_price)
        elif selected_method == "Simulate":
            simulate_model(baseline_price)

        
    except ValueError:
        messagebox.showerror("Error", "Invalid baseline price. Please enter a valid number.")


# Load your data and set up the environment and agent
file_path = './Shopee-Product-20oz-September-2023-FINAL.xlsx'
raw_data = load_dataset(file_path)
processed_data, daily_demand = predict_daily_demand(raw_data)
daily = calculate_average_customers(processed_data)

columns_to_exclude = ['day', 'product_id']
columns_to_scale = [col for col in processed_data.columns if col not in columns_to_exclude]
subset_to_scale = processed_data[columns_to_scale]

scaler = MinMaxScaler()
scaled_subset = scaler.fit_transform(subset_to_scale)
scaled_df = pd.DataFrame(scaled_subset, columns=columns_to_scale)

scaled_data = pd.concat([processed_data[columns_to_exclude], scaled_df], axis=1)

clipped_features =7

# Create the main UI
root = tk.Tk()
root.title("RL-DPR")

# Set the size of the root window
root.geometry("400x300")

# Center the window on the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - 400) // 2
y_coordinate = (screen_height - 300) // 2
root.geometry(f"300x150+{x_coordinate}+{y_coordinate}")

# Entry page
label_intro = ttk.Label(root, text="Dynamic Pricing Model with RL!", font=("Helvetica", 12, "bold"))
label_intro.pack(pady=(30,10))


# Buttons for Train and Simulate
train_button = ttk.Button(root, text="Train", command=partial(get_baseline_price_and_run, "Train"))
train_button.pack(side=tk.LEFT, padx=(50,10), pady=10)

simulate_button = ttk.Button(root, text="Simulate", command=partial(get_baseline_price_and_run, "Simulate"))
simulate_button.pack(side=tk.RIGHT, padx=(10,50), pady=10)

root.mainloop()