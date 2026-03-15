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
        "id": f"synthetic_{random.randint(1000, 9999)}",
        "outcome": info.get("pegs", 32),
        "timestamp": "synthetic",
        "moves": trajectory
    }

if __name__ == "__main__":
    print("Generating 5 synthetic random trajectories...")
    samples = [generate_random_trajectory() for _ in range(5)]
    for i, s in enumerate(samples):
        print(f"Game {i+1} finished with {s['outcome']} pegs remaining in {len(s['moves'])} moves.")
