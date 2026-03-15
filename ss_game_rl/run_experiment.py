import argparse
import logging
import os
from pathlib import Path

import torch

import ss_game_rl.evaluate as evaluate_module
import ss_game_rl.export_onnx as export_module
import ss_game_rl.fetch_data as fetch_module
import ss_game_rl.ope_metrics as ope_module
import ss_game_rl.train as train


def parse_args():
    current_dir = Path(__file__).parent
    
    parser = argparse.ArgumentParser(description='Offline RL experiment orchestration for Peg Solitaire.')
    parser.add_argument('--human-data', default=str(current_dir / 'human_data.json'), help='Path to real human offline trajectories.')
    parser.add_argument('--synthetic-data', default=str(current_dir / 'synthetic_data.json'), help='Path to synthetic generated offline trajectories.')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--win-reward', type=float, default=100.0)
    parser.add_argument('--loss-penalty-multiplier', type=float, default=2.0)
    parser.add_argument('--step-penalty', type=float, default=-1.0)
    parser.add_argument('--algorithm', type=str, choices=[alg.value for alg in train.Algorithm], default=train.Algorithm.REINFORCE.value)
    parser.add_argument('--model-path', default=str(current_dir / 'peg_solitaire_policy.pth'))
    parser.add_argument('--onnx-path', default=str(current_dir.parent / 'public' / 'peg_solitaire_policy.onnx'))
    parser.add_argument('--eval-games', type=int, default=200)
    parser.add_argument('--skip-fetch', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--skip-ope', action='store_true')
    parser.add_argument('--skip-export', action='store_true')
    return parser.parse_args()


def ensure_dataset(args):
    human_path = Path(args.human_data)
    synthetic_path = Path(args.synthetic_data)
    
    # We only auto-fetch the human data. Synthetic data is assumed to be pre-generated.
    if not human_path.exists():
        if args.skip_fetch:
            logging.warning('Human dataset %s missing and --skip-fetch enabled.', human_path)
        else:
            fetch_module.fetch_trajectories(output_file=args.human_data)
            
    if not synthetic_path.exists():
        logging.warning('Synthetic dataset %s missing. Training may be suboptimal without perfect examples. Run environment.py to generate.', synthetic_path)
        
    return human_path.exists() or synthetic_path.exists()


def run_training(args):
    algorithm = train.Algorithm(args.algorithm)
    model = train.train_model(
        data_path=[args.human_data, args.synthetic_data],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        algorithm=algorithm,
        gamma=args.gamma,
        win_reward=args.win_reward,
        loss_penalty_multiplier=args.loss_penalty_multiplier,
        step_penalty=args.step_penalty,
    )
    if model is None:
        return None
    torch.save(model.state_dict(), args.model_path)
    logging.info('Saved model weights to %s', args.model_path)
    return model


def run_evaluation(args):
    logging.info('Running inpatient evaluation for %s', args.model_path)
    evaluate_module.evaluate_model(model_path=args.model_path, num_games=args.eval_games)


def run_ope(args):
    logging.info('Running off-policy evaluation against real human data: %s', args.human_data)
    ope_module.evaluate_ope(trajectories_file=args.human_data, model_path=args.model_path, gamma=args.gamma)


def run_export(args):
    logging.info('Exporting ONNX model to %s', args.onnx_path)
    export_module.export_to_onnx(model_path=args.model_path, output_path=args.onnx_path)


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    args = parse_args()

    if not ensure_dataset(args):
        return

    model = run_training(args)
    if model is None:
        logging.error('Training aborted; skipping remaining steps.')
        return

    if not args.skip_eval:
        run_evaluation(args)

    if not args.skip_ope:
        run_ope(args)

    if not args.skip_export:
        run_export(args)


if __name__ == '__main__':
    main()
