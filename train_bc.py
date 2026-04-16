import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- データ読み込み ---
data = np.load("data.npy", allow_pickle=True)
observations = np.array([d[0] for d in data], dtype=np.float32)
actions = np.array([d[1] for d in data], dtype=np.float32)

obs_tensor = torch.tensor(observations)
act_tensor = torch.tensor(actions)

dataset = TensorDataset(obs_tensor, act_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# --- ポリシーネットワーク ---
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return self.net(x)

obs_dim = observations.shape[1]
act_dim = actions.shape[1]
policy = Policy(obs_dim, act_dim)

# --- 学習 ---
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(50):
    total_loss = 0
    for obs_batch, act_batch in loader:
        pred = policy(obs_batch)
        loss = loss_fn(pred, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1:3d} | Loss: {total_loss / len(loader):.4f}")

torch.save(policy.state_dict(), "policy.pth")
print("Saved: policy.pth")
