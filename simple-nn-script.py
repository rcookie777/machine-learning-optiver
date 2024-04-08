import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# load the dataset
df = pd.read_csv("/mnt/scratch/tairaeli/cse_dat/train.csv")
df = df.dropna()
df["target_change"] = np.ones_like(df["target"])

df.loc[df["target"]<0,'target_change'] = 0

df = df.drop(["target","row_id"], axis=1)

# trying a thing
df = df.drop(["stock_id","time_id","date_id"], axis = 1)

df.head()

scaler = StandardScaler()
X = scaler.fit_transform(df.drop("target_change", axis=1))
y = df["target_change"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X_train_tensor.shape, y_train_tensor.shape

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Assuming your data has 12 features based on the dropped columns
        self.fc1 = nn.Linear(12, 64)  # Input layer to hidden layer 1
        self.fc2 = nn.Linear(64, 32)  # Hidden layer 1 to hidden layer 2
        self.fc3 = nn.Linear(32, 2)   # Hidden layer 2 to output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation function for hidden layer 1
        x = F.relu(self.fc2(x))  # Activation function for hidden layer 2
        x = self.fc3(x)  # No activation for the output layer
        return x

net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("Beginning Training", flush=True)
for epoch in range(100):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

print('Finished Training', flush=True)

def predict(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs).squeeze()
            predicted = (outputs > 0.5).float()  # Using 0.5 as the threshold
            predictions.extend(predicted.numpy())
            actuals.extend(labels.numpy())
    return actuals, predictions

# Predict on the test set
y_true, y_pred = predict(net, test_loader)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)  # This requires the probability scores, so adjust accordingly

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")
print(confusion_matrix(y_true, y_pred))