import random
from matplotlib import pyplot as plt 
import torch
import numpy as np
from data_prep import scale_aggregated_state
from plot import display_prices_graph

def simulate(env, agent, save_frequency = 10, save_path = 'model/saved/DP_model.pth'):
    num_episodes = 1
    for episode in range(num_episodes):
        # Reset the environment for a new episode
        aggregated_state = env.reset()  # Assume reset provides the aggregated state for the first day
        total_reward = 0
        done = False 
        d_prices = []
        d_revenues = []
        s_revenues = []
        demands = []
        for day in range(1, env.total_days + 1):  # Start from the second day, as the first day's state is already set
            # Choose an action based on the aggregated state
            action = agent.choose_action(aggregated_state)
            # print(f"action: {action}")
            # Take the chosen action and observe the next state, reward, and whether the episode is done
            next_aggregated_state, reward, done, d_revenue, s_revenue, demand = env.step(action, day)
            
            # Train the agent using the collected information
            agent.train(aggregated_state, action, reward, next_aggregated_state, done)

            # Update the aggregated state for the next day
            aggregated_state = next_aggregated_state

            total_reward += reward

            d_prices.append(action)
            d_revenues.append(d_revenue)
            s_revenues.append(s_revenue)
            demands.append(demand)

            if done:
                break

        # Optionally, log or print the total reward for the episode
        print(f"\n\nEpisode {episode + 1}, Total Reward: {total_reward}\n\n")
        if (episode + 1) % save_frequency == 0:
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved at episode {episode + 1}")
    

    print("\n\nDynamic Pricing Breakdown\n")
    for day, (d_price, d_revenue, demand) in enumerate(zip(d_prices, d_revenues, demands), start=1):
        print(f"Day {day} Price: {d_price:.2f} Demand: {demand:.2f} Revenue: {d_revenue:.2f}")


    print("\nStatic Pricing Breakdown\n")
    s_prices = [env.initial_price] * 30
    for day, (s_price, s_revenue, demand) in enumerate(zip(s_prices, s_revenues, demands), start=1):
        print(f"Day {day} Price: {s_price:.2f} Demand: {demand:.2f} Revenue: {s_revenue:.2f}")

    print("\n\nPricing Evaluation\n\n")

    # Calculate for total for both Dynamic and Static Revenues
    d_revenues_sum = sum(d_revenues)
    s_revenues_sum = sum(s_revenues)

    print(f"Total Revenue (Dynamic) {d_revenues_sum:.2f}")
    print(f"Total Revenue (Static) {s_revenues_sum:.2f}\n")

    # Dynamic Pricing
    dynamic_highest_revenue_day = max(enumerate(d_revenues, start=1), key=lambda x: x[1])[0]
    dynamic_lowest_revenue_day = min(enumerate(d_revenues, start=1), key=lambda x: x[1])[0]

    print("Dynamic Pricing Results:")
    print(f"Highest Revenue Day: Day {dynamic_highest_revenue_day}")
    print(f"Price: {d_prices[dynamic_highest_revenue_day - 1]:.2f}, Demand: {demands[dynamic_highest_revenue_day - 1]}, Revenue: {d_revenues[dynamic_highest_revenue_day - 1]:.2f}")

    print(f"\nLowest Revenue Day: Day {dynamic_lowest_revenue_day}")
    print(f"Price: {d_prices[dynamic_lowest_revenue_day - 1]:.2f}, Demand: {demands[dynamic_lowest_revenue_day - 1]}, Revenue: {d_revenues[dynamic_lowest_revenue_day - 1]:.2f}")

    # Static Pricing
    static_highest_revenue_day = max(enumerate(s_revenues, start=1), key=lambda x: x[1])[0]
    static_lowest_revenue_day = min(enumerate(s_revenues, start=1), key=lambda x: x[1])[0]
    
    print("\nStatic Pricing Results:")
    print(f"Highest Revenue Day: Day {static_highest_revenue_day}")
    print(f"Price: {s_prices[static_highest_revenue_day - 1]:.2f}, Demand: {demands[static_highest_revenue_day - 1]:.2f}, Revenue: {s_revenues[static_highest_revenue_day - 1]:.2f}")

    print(f"\nLowest Revenue Day: Day {static_lowest_revenue_day}")
    print(f"Price: {s_prices[static_lowest_revenue_day - 1]:.2f}, Demand: {demands[static_lowest_revenue_day - 1]:.2f}, Revenue: {s_revenues[static_lowest_revenue_day - 1]:.2f}")
    
    print("\n\n____________________________________________\n\n")
    print("Analysis\n\n")
    # Calculate mean for both Dynamic and Static Pricing
    mean_dynamic = sum(d_revenues) / len(d_revenues)
    mean_static = sum(s_revenues) / len(s_revenues)

    print(f"Mean for Dynamic Pricing: {mean_dynamic:.2f}")
    print(f"Mean for Static Pricing: {mean_static:.2f}\n")

    # Calculate variance and standard deviation for both Dynamic and Static Pricing
    variance_dynamic = sum((x - mean_dynamic) ** 2 for x in d_revenues) / len(d_revenues)
    std_dev_dynamic = variance_dynamic ** 0.5

    variance_static = sum((x - mean_static) ** 2 for x in s_revenues) / len(s_revenues)
    std_dev_static = variance_static ** 0.5

    print(f"Standard Deviation for Dynamic Pricing: {std_dev_dynamic:.2f}")
    print(f"Standard Deviation for Static Pricing: {std_dev_static:.2f}\n")

    # Calculate margin of error (assuming 95% confidence level)
    num_samples = len(d_revenues)
    margin_error_dynamic = 1.96 * (std_dev_dynamic / (num_samples ** 0.5))
    margin_error_static = 1.96 * (std_dev_static / (num_samples ** 0.5))

    print(f"Margin of Error for Dynamic Pricing: {margin_error_dynamic:.2f}")
    print(f"Margin of Error for Static Pricing: {margin_error_static:.2f}\n")
    
    from sklearn.linear_model import LinearRegression

    # Convert lists to NumPy arrays
    demand = np.array(demands)
    price = np.array(d_prices)

    demand = demand.reshape(-1, 1)  # Reshape to a column vector
    price = price.reshape(-1, 1)  # Reshape to a column vector

    # Step 1: Predict Demand for the 31st day
    model_demand = LinearRegression()
    model_demand.fit(np.arange(1, len(demand) + 1).reshape(-1, 1), demand)  # Fit the model

    print("\n\n____________________________________________\n\n")
    print("Predicction\n\n")
    predicted_demand_31 = model_demand.predict(np.array([[31]]))  # Predict the demand for day 31
    predicted_demand_31 = predicted_demand_31[0][0]  # Extracting the predicted demand value
    print(f"Predicted demand for the 31st day: {predicted_demand_31:.2f}")

    # Step 2: Predict Price for the Predicted Demand
    model_price = LinearRegression()
    model_price.fit(demand, price)  # Fit the model

    predicted_price_31 = model_price.predict(np.array([[predicted_demand_31]]))  # Predict price for the predicted demand
    predicted_price_31 = predicted_price_31[0][0]  # Extracting the predicted price value
    predicted_price_31 += 46
    print(f"Recommended price for the 31st day based on demand: {predicted_price_31:.2f}\n\n")
    
    display_prices_graph(d_prices, s_prices, d_revenues, s_revenues,
        demands, predicted_price_31, predicted_demand_31)
