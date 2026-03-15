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
    def __init__(self, trajectories_file):
        """
        Loads data. Expects a JSON file with a list of game trajectories:
        { "outcome": X, "moves": [ {"state": "...", "action": {"row": r, "col": c, "dir": d}} ] }
        """
        self.transitions = []
        if os.path.exists(trajectories_file):
            with open(trajectories_file, 'r') as f:
                data = json.load(f)
                
            for game in data:
                # Basic Behavioral Cloning: Just learn the actions humans took.
                # In true Offline RL (like CQL or AWAC), you would weight these by the 'outcome'.
                outcome = game.get('outcome', 32)
                weight = 1.0 / (outcome + 1) # Prefer games that ended with fewer pegs
                
                for move in game['moves']:
                    state = board_to_tensor(move['state'])
                    
                    # Convert action (r, c, dir) into a single integer class (0 to 195)
                    # 7 * 7 * 4 = 196 possible moves total
                    r = move['action']['row']
                    c = move['action']['col']
                    d = move['action']['dir']
                    action_idx = (r * 7 * 4) + (c * 4) + d
                    
                    self.transitions.append((state, action_idx, weight))
        else:
            print(f"Warning: {trajectories_file} not found. Run fetch_data.py first.")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state, action, weight = self.transitions[idx]
        return state, torch.tensor(action, dtype=torch.long), torch.tensor(weight, dtype=torch.float32)


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
    # Using CrossEntropyLoss for behavioral cloning (classification task)
    criterion = nn.CrossEntropyLoss(reduction='none') 
    
    print(f"Starting training on {len(dataset)} state-action pairs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for states, actions, weights in dataloader:
            optimizer.zero_grad()
            
            logits = model(states)
            
            # Unweighted loss: loss = criterion(logits, actions)
            # Weighted loss based on the game's final outcome:
            unweighted_loss = criterion(logits, actions)
            loss = (unweighted_loss * weights).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")
        
    print("Training complete! Saving model...")
    torch.save(model.state_dict(), "peg_solitaire_policy.pth")
    print("Saved to peg_solitaire_policy.pth")
    
    # Future step: Export to ONNX here to upload to Vercel
    # torch.onnx.export(model, torch.randn(1, 1, 7, 7), "peg_solitaire_policy.onnx")

if __name__ == "__main__":
    train_model()
