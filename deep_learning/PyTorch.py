# Import the required packages
import torch
from torch import nn
from torch.utils.data import DataLoader


class TabularModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(TabularModel, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_layers[i], hidden_layers[i + 1]), nn.ReLU()) for i in range(len(hidden_layers) - 1)],
            nn.Linear(hidden_layers[-1], output_dim)
        )

    def forward(self, x):
        return self.fc_layers(x)

# Instantiate the model
model = TabularModel(input_dim=X_train.shape[1], output_dim=1, hidden_layers=[100, 50])

criterion = nn.MSELoss()  # For regression tasks
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs.squeeze(), batch_labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

model.eval()
total_loss = 0
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        loss = criterion(outputs.squeeze(), batch_labels)
        total_loss += loss.item()

print(f'Test Loss: {total_loss / len(test_loader)}')
