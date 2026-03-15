# Peg Solitaire Offline RL Implementation Plan

## Objective
Train a Neural Network (Policy Network) to play Peg Solitaire perfectly, utilizing human gameplay data collected from the Next.js frontend and synthetic data generated locally. Eventually, deploy this model to the frontend to replace the Depth-First Search (DFS) Hint Engine for instant, zero-cost AI inference.

## Current State (Stage 1 Complete)
- **Data Collection (Web)**: The Next.js app formats state-action pairs into the RL format `(row, col, direction)` and pushes trajectories to Upstash Redis (`rl:trajectories`).
- **Data Ingestion (Local)**: `fetch_data.py` successfully connects to Upstash and pulls the data to `trajectories.json`.
- **Simulation**: `environment.py` provides a NumPy implementation of the game for synthetic data generation.
- **Model Definition**: `train.py` defines a lightweight CNN Policy Network tailored for the 7x7 board and 196 possible action classes.

## Next Steps

### Phase 1: Data Augmentation & Baseline Training
1. **Gather Human Data**: Play multiple games on the deployed Vercel application to populate the Redis list with human trajectories.
2. **Generate Synthetic Data**: Expand `environment.py` to use a more intelligent rollout (like Monte Carlo Tree Search) instead of purely random actions, generating high-quality synthetic winning trajectories.
3. **Train Baseline Model**: Execute `train.py` using Behavioral Cloning (CrossEntropyLoss scaled by outcome). Validate the model's loss curve decreases over epochs.

### Phase 2: Advanced Offline RL (Optional)
If Behavioral Cloning solely copies human mistakes (since humans often lose Peg Solitaire), implement true Offline RL algorithms:
1. **Advantage Weighted Actor-Critic (AWAC) / Conservative Q-Learning (CQL)**: Modify `train.py` to implement an actor-critic architecture that learns from suboptimal data by penalizing out-of-distribution actions.

### Phase 3: Model Evaluation
1. **Create `evaluate.py`**: Write a script that loads `peg_solitaire_policy.pth`, runs it against the `PegSolitaireEnv` for 1000 games, and calculates the win rate (percentage of games ending with 1 peg in the center).

### Phase 4: Production Deployment (Inference)
1. **Export to ONNX**: Create `export_onnx.py` to convert the trained PyTorch model (`.pth`) into the standardized ONNX format (`.onnx`).
2. **Web Integration**:
    - Add `onnxruntime-web` to the Next.js project.
    - Place the `.onnx` model file in the Next.js `public/` directory.
    - Modify the "Get Hint" button logic in `GameBoard.tsx`: Instead of calling the `/api/hint` DFS backend, download the ONNX model to the browser and run inference locally in WebAssembly. This provides instant hints with zero server cost.
