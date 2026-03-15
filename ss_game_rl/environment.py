import random
import numpy as np

class PegSolitaireEnv:
    """
    A lightweight Peg Solitaire environment for Python RL.
    Board shape is 7x7.
    States are represented as 49-character strings (like Vercel) or NumPy arrays.
    Actions are (row, col, direction).
    """
    DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)] # Right, Down, Left, Up

    def __init__(self):
        self.reset()

    def reset(self):
        # 1 = Peg, 0 = Empty, -1 = Blocked (corners)
        self.board = np.array([
            [-1, -1,  1,  1,  1, -1, -1],
            [-1, -1,  1,  1,  1, -1, -1],
            [ 1,  1,  1,  1,  1,  1,  1],
            [ 1,  1,  1,  0,  1,  1,  1],
            [ 1,  1,  1,  1,  1,  1,  1],
            [-1, -1,  1,  1,  1, -1, -1],
            [-1, -1,  1,  1,  1, -1, -1]
        ], dtype=np.int8)
        self.done = False
        return self._get_state_str()

    def step(self, action):
        """
        Action is a tuple (row, col, direction)
        direction: 0=Right, 1=Down, 2=Left, 3=Up
        """
        assert not self.done, "Game is already over"
        r, c, d = action
        dr, dc = self.DIRS[d]
        
        # Validate move (in training, you would mask invalid moves entirely)
        if not self._is_valid_move(r, c, d):
            return self._get_state_str(), -10, True, {"msg": "Invalid move"}

        # Apply move
        self.board[r, c] = 0
        self.board[r + dr, c + dc] = 0
        self.board[r + 2*dr, c + 2*dc] = 1

        moves_left = self.get_legal_actions()
        self.done = len(moves_left) == 0

        pegs_remaining = np.sum(self.board == 1)
        
        # Reward shaping: Small penalty for taking a step, big reward for winning
        reward = -1 
        if self.done:
            if pegs_remaining == 1:
                if self.board[3, 3] == 1:
                    reward += 100 # Perfect Win
                else:
                    reward += 50  # Win, but not in center
            else:
                reward -= (pegs_remaining * 2) # Penalty for pegs left

        return self._get_state_str(), reward, self.done, {"pegs": pegs_remaining}

    def get_legal_actions(self):
        moves = []
        for r in range(7):
            for c in range(7):
                if self.board[r, c] == 1:
                    for d in range(4):
                        if self._is_valid_move(r, c, d):
                            moves.append((r, c, d))
        return moves

    def _is_valid_move(self, r, c, d):
        dr, dc = self.DIRS[d]
        r1, c1 = r + dr, c + dc
        r2, c2 = r + 2*dr, c + 2*dc
        
        if r2 < 0 or r2 >= 7 or c2 < 0 or c2 >= 7:
            return False
            
        return (self.board[r, c] == 1 and 
                self.board[r1, c1] == 1 and 
                self.board[r2, c2] == 0)

    def _get_state_str(self):
        # Convert numpy array to Vercel's JSON format ('O', '.', ' ')
        char_map = {-1: ' ', 0: '.', 1: 'O'}
        return "".join([char_map[v] for v in self.board.flatten()])

def generate_random_trajectory():
    """Generates a synthetic trajectory by taking random valid moves."""
    env = PegSolitaireEnv()
    state = env.reset()
    
    trajectory = []
    
    while not env.done:
        valid_moves = env.get_legal_actions()
        if not valid_moves:
            break
            
        action = random.choice(valid_moves)
        r, c, d = action
        trajectory.append({
            "state": state,
            "action": {"row": r, "col": c, "dir": d}
        })
        
        next_state, reward, done, info = env.step(action)
        state = next_state
        
    return {
        "id": f"synthetic_random_{random.randint(10000, 99999)}",
        "outcome": info.get("pegs", 32),
        "timestamp": "synthetic",
        "moves": trajectory
    }

def generate_perfect_trajectory():
    """
    Uses a Depth-First Search to find a perfect winning game (1 peg left in center).
    Since true DFS from the start is too slow, we play random moves until 15 pegs are left, 
    then use DFS to perfectly finish the game.
    """
    env = PegSolitaireEnv()
    env.reset()
    
    trajectory = []
    
    # 1. Random rollout until 14 pegs are left (to create diverse board states)
    while np.sum(env.board == 1) > 14 and not env.done:
        moves = env.get_legal_actions()
        if not moves: break
        
        # Heuristic: Prefer moves towards the center to keep the board solvable
        center_moves = []
        for a in moves:
            r, c, d = a
            dr, dc = env.DIRS[d]
            r2, c2 = r + 2*dr, c + 2*dc
            if abs(r2 - 3) <= 2 and abs(c2 - 3) <= 2:
                center_moves.append(a)
                
        action = random.choice(center_moves) if center_moves else random.choice(moves)
        
        state = env._get_state_str()
        trajectory.append({"state": state, "action": {"row": action[0], "col": action[1], "dir": action[2]}})
        env.step(action)
        
    # If the random rollout got stuck, just return it as a losing game
    if env.done and np.sum(env.board == 1) > 1:
        return {
            "id": f"synthetic_loss_{random.randint(10000, 99999)}",
            "outcome": int(np.sum(env.board == 1)),
            "timestamp": "synthetic",
            "moves": trajectory
        }

    # 2. DFS from the 14-peg state to find the perfect ending
    def dfs(current_env, path, depth_budget):
        if depth_budget <= 0: return None
        if np.sum(current_env.board == 1) == 1:
            if current_env.board[3, 3] == 1:
                return path # Found perfect win!
            return None # Won, but not in center
            
        moves = current_env.get_legal_actions()
        if not moves: return None
        
        for move in moves:
            # Clone env efficiently
            new_env = PegSolitaireEnv()
            new_env.board = current_env.board.copy()
            new_env.done = current_env.done
            
            state_str = new_env._get_state_str()
            new_env.step(move)
            
            result = dfs(new_env, path + [{"state": state_str, "action": {"row": move[0], "col": move[1], "dir": move[2]}}], depth_budget - 1)
            if result is not None:
                return result
                
        return None

    # Budget DFS so python doesn't hang forever on unsolvable random rollouts
    winning_path = dfs(env, [], 3000)
    
    if winning_path:
        # Combine the random opening with the perfect ending
        full_trajectory = trajectory + winning_path
        return {
            "id": f"synthetic_perfect_{random.randint(10000, 99999)}",
            "outcome": 1,
            "timestamp": "synthetic",
            "moves": full_trajectory
        }
    else:
        # The random opening led to an unsolvable state
        return None

if __name__ == "__main__":
    import json
    import os
    
    print("Generating synthetic trajectories...")
    samples = []
    
    # Generate perfect games using randomized rollouts + DFS solvers
    print("Generating perfect wins via DFS (this will take 1-2 minutes)...")
    wins_needed = 20
    attempts = 0
    while len(samples) < wins_needed:
        attempts += 1
        traj = generate_perfect_trajectory()
        if traj and traj['outcome'] == 1:
            samples.append(traj)
            print(f"  Found perfect game! ({len(samples)}/{wins_needed}) in {attempts} random DFS rollouts.")
            attempts = 0 # reset
            
    # Generate random losing games to teach the network what NOT to do
    print("Generating 80 random losing games...")
    for _ in range(80):
        samples.append(generate_random_trajectory())
        
    output_file = "synthetic_data.json"
    
    # If trajectories.json exists, we append to it.
    if os.path.exists("trajectories.json"):
        with open("trajectories.json", "r") as f:
            existing = json.load(f)
        total_samples = existing + samples
        output_file = "trajectories.json"
    else:
        total_samples = samples
        
    with open(output_file, 'w') as f:
        json.dump(total_samples, f, indent=2)
        
    print(f"Saved {len(total_samples)} total trajectories to {output_file}")
