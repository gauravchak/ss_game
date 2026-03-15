import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Map Vercel string format to a numeric tensor representation
def board_to_tensor(board_str):
    mapping = {' ': -1, '.': 0, 'O': 1}
    # board_str is 49 chars long
    arr = [mapping[c] for c in board_str]
    return torch.tensor(arr, dtype=torch.float32).view(1, 7, 7) # (Channels, Height, Width)

class PegSolitaireDataset(Dataset):
    def __init__(self, trajectories_file, gamma=0.99):
        """
        Loads data and computes discounted rewards for Policy Gradient.
        """
        self.transitions = []
        if os.path.exists(trajectories_file):
            with open(trajectories_file, 'r') as f:
                data = json.load(f)
                
            for game in data:
                outcome = game.get('outcome', 32)
                
                # Base Reward Formulation:
                # We want a high positive reward for winning (1 peg)
                # and negative rewards for losing.
                if outcome == 1:
                    final_reward = 100.0
                else:
                    final_reward = -float(outcome * 2)
                    
                moves = game['moves']
                returns = []
                
                # Calculate Discounted Returns (backwards)
                R = final_reward
                for _ in reversed(moves):
                    returns.insert(0, R)
                    R = -1.0 + gamma * R # Small step penalty (-1) + discounted future reward
                    
                # Normalize returns for training stability (standard Policy Gradient trick)
                returns = np.array(returns)
                if len(returns) > 1 and returns.std() > 0:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
                for i, move in enumerate(moves):
                    state = board_to_tensor(move['state'])
                    
                    r = move['action']['row']
                    c = move['action']['col']
                    d = move['action']['dir']
                    action_idx = (r * 7 * 4) + (c * 4) + d
                    
                    self.transitions.append((state, action_idx, returns[i]))
        else:
            print(f"Warning: {trajectories_file} not found. Run fetch_data.py first.")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state, action, discounted_reward = self.transitions[idx]
        return state, torch.tensor(action, dtype=torch.long), torch.tensor(discounted_reward, dtype=torch.float32)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # A lightweight CNN since the board is a 2D grid structure
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # 64 channels * 7 height * 7 width = 3136
        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 196) # Output 196 raw action logits (7x7 origin * 4 directions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        logits = self.fc_layers(x)
        return logits


def train_model(data_path="trajectories.json", epochs=50, batch_size=32):
    dataset = PegSolitaireDataset(data_path)
    if len(dataset) == 0:
        print("Dataset is empty. Cannot train.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Starting Policy Gradient training on {len(dataset)} state-action pairs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for states, actions, rewards in dataloader:
            optimizer.zero_grad()
            
            # 1. Forward pass: Get raw logits for all 196 possible actions
            logits = model(states)
            
            # 2. Get probabilities using Softmax
            probs = torch.softmax(logits, dim=1)
            
            # 3. Gather the predicted probabilities of the actions that were ACTUALLY taken in the trajectory
            # Gather expects indices of shape (batch, 1), so we unsqueeze(1) and then squeeze() the result
            action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # 4. Calculate REINFORCE Loss: -log(P(action)) * Reward
            # We want to maximize the probability of actions that led to high rewards.
            # PyTorch minimizes loss, so we use negative log probability.
            log_probs = torch.log(action_probs + 1e-10) # Add epsilon to prevent log(0)
            
            loss = -(log_probs * rewards).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - PG Loss: {total_loss/len(dataloader):.4f}")
        
    print("Training complete! Saving model...")
    torch.save(model.state_dict(), "peg_solitaire_policy.pth")
    print("Saved to peg_solitaire_policy.pth")
    
    # Future step: Export to ONNX here to upload to Vercel
    # torch.onnx.export(model, torch.randn(1, 1, 7, 7), "peg_solitaire_policy.onnx")

if __name__ == "__main__":
    train_model()
