import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load Fantasy Premier League data
# Replace 'your_dataset.csv' with the actual file containing your FPL data
data = pd.read_csv('your_dataset.csv')

# Starting features
feature_columns = ['player_id', 'gameweek', 'previous_points', 'minutes_played', 'goals_scored', 'assists']

# Select relevant features and target
data = data[feature_columns + ['total_points']]

# Sort the data by player_id and gameweek
data = data.sort_values(by=['player_id', 'gameweek'])

# Create sequences of data for each player
sequence_length = 5  # Adjust as needed
sequences = []
for player_id, group in data.groupby('player_id'):
    player_sequence = [group.iloc[i:i + sequence_length] for i in range(len(group) - sequence_length + 1)]
    sequences.extend(player_sequence)

# Convert sequences to numpy arrays
sequences = [np.array(sequence) for sequence in sequences]

# Split data into train and test sets
train_data, test_data = train_test_split(sequences, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
def prepare_sequences(data):
    X = torch.tensor([sequence[:, 2:].astype(np.float32) for sequence in data])  # Exclude player_id and gameweek
    y = torch.tensor([sequence[-1, 2] for sequence in data])  # Target is the 'total_points' of the last gameweek
    return X, y

X_train, y_train = prepare_sequences(train_data)
X_test, y_test = prepare_sequences(test_data)

# Define a simple LSTM model
class FPLForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(FPLForecastModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use only the last time step's output for forecasting
        return out

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.view(-1, X_train.size(-1))).view(X_train.size())
X_test_scaled = scaler.transform(X_test.view(-1, X_test.size(-1))).view(X_test.size())

# Hyperparameters
input_size = X_train.size(-1)
hidden_size = 64
num_layers = 2
output_size = 1
learning_rate = 0.001
num_epochs = 50

# Create DataLoader for training
train_data = TensorDataset(X_train_scaled, y_train)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Initialize the model, loss function, and optimizer
model = FPLForecastModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_inputs = torch.tensor(X_test_scaled, dtype=torch.float32)
    predictions = model(test_inputs).squeeze().numpy()

# Compare predictions with actual scores
for pred, actual in zip(predictions, y_test.numpy()):
    print(f'Predicted: {pred:.2f}, Actual: {actual:.2f}')
