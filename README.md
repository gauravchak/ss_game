# Aurelian Solitaire + Offline RL Research

This repository hosts the **Aurelian Solitaire** experience: a stylized peg-solitaire challenge in Next.js with confetti rewards, global leaderboards, and an embedded hint system powered by ONNX inference inside the browser. Every game played on the deployed web app (https://ss-game-three.vercel.app/) streams trajectories (state-action pairs) to Upstash Redis so they can be re-used in offline reinforcement learning experiments.

## Play the Demo

- Visit https://ss-game-three.vercel.app/ to play the fully functional UI built in `src/components/GameBoard.tsx` plus the leaderboard in `src/components/Leaderboard.tsx`.
- Every completed game (wins, losses, and everything in between) pushes its trajectory to `rl:trajectories` inside Upstash. That log becomes the single replay buffer for the offline RL experiments.

## Research Objective

The goal is to compare off-policy policy-gradient algorithms (starting with REINFORCE and later GRPO or similar candidates) trained only on logged data — no live rollout, only the human + synthetic trajectories collected in `ss_game_rl/trajectories.json`. The experiments aim to answer: which offline policy-gradient approach converges to a better peg-solitaire policy when replaying fixed behavior data?

## Off-policy REINFORCE Primer

- Trajectories saved by the browser consist of a 49-character board string plus the chosen `(row, col, direction)` action and the final marble count.
- `ss_game_rl/train.py` replays those logs, constructs tensors via `board_to_tensor()`, and computes normalized returns that backpropagate the final reward signal (currently +100 for a perfect single-peg finish and -2 per remaining peg).
- The training loop treats these returns as the per-timestep objective, divides by the batch standard deviation to keep gradients stable, and updates the shared policy network — a classic REINFORCE update applied to off-policy, logged data rather than freshly collected rollouts.
- This setup makes it straightforward to swap in alternative policy-gradient critics (GRPO, AWAC, etc.) later and compare their simulator win rates plus OPE estimates while keeping the replay buffer constant.
