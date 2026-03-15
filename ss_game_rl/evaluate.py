import torch
from train import PolicyNetwork, board_to_tensor
from environment import PegSolitaireEnv

def evaluate_model(model_path="peg_solitaire_policy.pth", num_games=100):
    env = PegSolitaireEnv()
    model = PolicyNetwork()
    
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    wins = 0
    total_pegs_left = 0
    center_wins = 0

    print(f"Playing {num_games} games...")
    
    with torch.no_grad():
        for i in range(num_games):
            state_str = env.reset()
            
            while not env.done:
                legal_moves = env.get_legal_actions()
                if not legal_moves:
                    break
                
                # Convert board state to tensor
                state_tensor = board_to_tensor(state_str).unsqueeze(0) # Add batch dimension
                
                # Get model predictions
                logits = model(state_tensor)
                
                # In RL, we only want to pick from LEGAL moves
                # So we apply a heavy penalty (mask) to illegal moves before taking argmax
                mask = torch.full((1, 196), -float('inf'))
                
                for r, c, d in legal_moves:
                    action_idx = (r * 7 * 4) + (c * 4) + d
                    mask[0, action_idx] = 0.0 # Allow this index
                    
                # Add mask to logits (illegal moves become -inf)
                masked_logits = logits + mask
                
                # Determine best move according to the policy
                best_action_idx = torch.argmax(masked_logits, dim=1).item()
                
                # Decode the index back to (r, c, dir)
                d = best_action_idx % 4
                c = (best_action_idx // 4) % 7
                r = (best_action_idx // 28) % 7
                
                action = (r, c, d)
                
                # Ensure the chosen action is actually legal
                if action not in legal_moves:
                    # Should theoretically never happen with the exact mask above
                    print("Model attempted an illegal move! Falling back to random.")
                    import random
                    action = random.choice(legal_moves)
                
                next_state, reward, done, info = env.step(action)
                state_str = next_state
                
            # Game over analysis
            pegs = info.get("pegs", 32)
            total_pegs_left += pegs
            
            if pegs == 1:
                wins += 1
                if env.board[3, 3] == 1:
                    center_wins += 1
                    
    print("\n--- Evaluation Results ---")
    print(f"Games Played: {num_games}")
    print(f"Total Wins (1 peg left): {wins} ({wins/num_games * 100:.1f}%)")
    print(f"Center Wins (Perfect): {center_wins} ({center_wins/num_games * 100:.1f}%)")
    print(f"Average Pegs Remaining: {total_pegs_left / num_games:.2f}")

if __name__ == "__main__":
    evaluate_model()
