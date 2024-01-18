import torch
import torch.nn as nn
import numpy as np

# Define a simple neural network model (replace with your model)
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Instantiate the model (replace input_size and output_size with actual values)
input_size = 10  # Replace with the actual input size of your aggregated state
output_size = 5  # Replace with the actual number of actions in your action space
model = SimpleModel(input_size, output_size)

# Define a function to calculate Q-values
def calculate_q_values(aggregated_state):
    state_tensor = torch.Tensor(aggregated_state)
    q_values = model(state_tensor)
    return q_values

# Example aggregated state (replace with your actual aggregated state)
aggregated_state_example = np.random.rand(input_size)

# Print Q-values for the example aggregated state
q_values_example = calculate_q_values(aggregated_state_example)
print(f"Aggregated State: {aggregated_state_example}")
print(f"Q-values: {q_values_example}")
