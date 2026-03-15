import argparse
import json
import logging
import os
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Map Vercel string format to a numeric tensor representation
def board_to_tensor(board_str):
    mapping = {' ': -1, '.': 0, 'O': 1}
    arr = [mapping[c] for c in board_str]
    return torch.tensor(arr, dtype=torch.float32).view(1, 7, 7)


class Algorithm(Enum):
    REINFORCE = 'reinforce'
    GRPO = 'grpo'

class PegSolitaireDataset(Dataset):
    def __init__(
        self,
        trajectories_files,
        gamma=0.99,
        win_reward=100.0,
        loss_penalty_multiplier=2.0,
        step_penalty=-1.0
    ):
        self.transitions = []
        self.gamma = gamma
        self.win_reward = win_reward
        self.loss_penalty_multiplier = loss_penalty_multiplier
        self.step_penalty = step_penalty

        if isinstance(trajectories_files, str):
            trajectories_files = [trajectories_files]

        data = []
        for file_path in trajectories_files:
            if not os.path.exists(file_path):
                logging.warning('%s not found - skipping', file_path)
                continue
            with open(file_path, 'r') as f:
                data.extend(json.load(f))

        for game in data:
            outcome = game.get('outcome', 32)
            final_reward = self._final_reward(outcome)

            moves = game.get('moves', [])
            if not moves:
                continue

            returns = self._compute_returns(len(moves), final_reward)

            for idx, move in enumerate(moves):
                state = board_to_tensor(move['state'])
                r = move['action']['row']
                c = move['action']['col']
                d = move['action']['dir']
                action_idx = (r * 7 * 4) + (c * 4) + d
                self.transitions.append((state, action_idx, returns[idx]))

    def _final_reward(self, outcome: int) -> float:
        if outcome == 1:
            return self.win_reward
        return -float(outcome * self.loss_penalty_multiplier)

    def _compute_returns(self, length: int, final_reward: float):
        returns = []
        R = final_reward
        for _ in range(length):
            returns.insert(0, R)
            R = self.step_penalty + self.gamma * R

        returns = np.array(returns, dtype=np.float32)
        if len(returns) > 1 and returns.std() > 0:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        state, action, discounted_reward = self.transitions[idx]
        return state, torch.tensor(action, dtype=torch.long), torch.tensor(discounted_reward, dtype=torch.float32)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 196)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)


def reinforce_training_step(model, optimizer, states, actions, rewards):
    optimizer.zero_grad()
    logits = model(states)
    probs = torch.softmax(logits, dim=1)
    action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    log_probs = torch.log(action_probs + 1e-10)
    loss = -(log_probs * rewards).mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def grpo_training_step(model, old_model, optimizer, states, actions, rewards, clip_epsilon=0.2, kl_coef=0.01):
    optimizer.zero_grad()
    
    # 1. Group Advantage (Z-Score Normalization across the batch)
    if rewards.std() > 0:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    else:
        advantages = rewards - rewards.mean()

    # 2. Current Policy Probabilities
    logits = model(states)
    probs = torch.softmax(logits, dim=1)
    action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    # 3. Old Policy Probabilities (No gradients)
    with torch.no_grad():
        old_logits = old_model(states)
        old_probs = torch.softmax(old_logits, dim=1)
        old_action_probs = old_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
    # 4. Probability Ratio
    ratio = action_probs / (old_action_probs + 1e-10)
    
    # 5. Clipped Surrogate Objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # 6. KL Divergence Penalty
    kl_div = torch.sum(old_probs * (torch.log(old_probs + 1e-10) - torch.log(probs + 1e-10)), dim=1).mean()
    
    loss = policy_loss + kl_coef * kl_div
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    optimizer.step()
    return loss.item(), kl_div.item()


def train_model(
    data_path=["human_data.json", "synthetic_data.json"],
    epochs=50,
    batch_size=32,
    lr=1e-3,
    algorithm=Algorithm.GRPO,
    gamma=0.99,
    win_reward=100.0,
    loss_penalty_multiplier=2.0,
    step_penalty=-1.0,
    clip_epsilon=0.2,
    kl_coef=0.01
):
    dataset = PegSolitaireDataset(
        data_path,
        gamma=gamma,
        win_reward=win_reward,
        loss_penalty_multiplier=loss_penalty_multiplier,
        step_penalty=step_penalty,
    )
    if len(dataset) == 0:
        logging.warning('Dataset empty (%s). Aborting training.', data_path)
        return None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = PolicyNetwork()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    old_model = None
    if algorithm == Algorithm.GRPO:
        old_model = PolicyNetwork()
        old_model.load_state_dict(model.state_dict())
        old_model.eval()

    logging.info('Training %s on %d samples', algorithm.value, len(dataset))

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_kl = 0.0
        
        for states, actions, rewards in dataloader:
            if algorithm == Algorithm.REINFORCE:
                epoch_loss += reinforce_training_step(model, optimizer, states, actions, rewards)
            elif algorithm == Algorithm.GRPO:
                loss, kl = grpo_training_step(
                    model, old_model, optimizer, states, actions, rewards,
                    clip_epsilon=clip_epsilon, kl_coef=kl_coef
                )
                epoch_loss += loss
                epoch_kl += kl
                
                # Soft-update old_model every batch to prevent the probability ratio from exploding
                with torch.no_grad():
                    for param, old_param in zip(model.parameters(), old_model.parameters()):
                        old_param.data.copy_(0.995 * old_param.data + 0.005 * param.data)
            else:
                raise NotImplementedError(f'Algorithm {algorithm} is not implemented yet')

        if algorithm == Algorithm.GRPO:
            logging.info('Epoch %d/%d - avg loss %.4f - KL Div %.4f', epoch + 1, epochs, epoch_loss / len(dataloader), epoch_kl / len(dataloader))
        else:
            logging.info('Epoch %d/%d - avg loss %.4f', epoch + 1, epochs, epoch_loss / len(dataloader))

    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train the Peg Solitaire policy off-policy.')
    parser.add_argument('--data', nargs='+', default=['human_data.json', 'synthetic_data.json'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--win-reward', type=float, default=100.0)
    parser.add_argument('--loss-penalty-multiplier', type=float, default=2.0)
    parser.add_argument('--step-penalty', type=float, default=-1.0)
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    parser.add_argument('--kl-coef', type=float, default=0.01)
    parser.add_argument('--algorithm', type=str, choices=[alg.value for alg in Algorithm], default=Algorithm.GRPO.value)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    args = parse_args()
    algorithm = Algorithm(args.algorithm)

    model = train_model(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        algorithm=algorithm,
        gamma=args.gamma,
        win_reward=args.win_reward,
        loss_penalty_multiplier=args.loss_penalty_multiplier,
        step_penalty=args.step_penalty,
        clip_epsilon=args.clip_epsilon,
        kl_coef=args.kl_coef
    )

    if model is None:
        return

    output_path = 'peg_solitaire_policy.pth'
    torch.save(model.state_dict(), output_path)
    logging.info('Saved trained model to %s', output_path)


if __name__ == '__main__':
    main()
