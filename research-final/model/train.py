
import torch

def training(env, agent, num_episodes = 50, save_frequency = 10, save_path = 'model/saved/DP_model.pth'):
    for episode in range(num_episodes):
        # Reset the environment for a new episode
        aggregated_state = env.reset()  # Assume reset provides the aggregated state for the first day
        total_reward = 0
        done = False 
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

            if done:
                break

        # Optionally, log or print the total reward for the episode
        print(f"\n\nEpisode {episode + 1}, Total Reward: {total_reward}\n\n")
        if (episode + 1) % save_frequency == 0:
            torch.save(agent.model.state_dict(), save_path)
            print(f"Model saved at episode {episode + 1}")

