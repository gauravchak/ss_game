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
    parser.add_argument('--data', default=str(current_dir / 'trajectories.json'), help='Path to the offline trajectory dataset.')
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
    data_path = Path(args.data)
    if data_path.exists():
        return True
    if args.skip_fetch:
        logging.error('Dataset %s missing and --skip-fetch enabled.', data_path)
        return False
    fetch_module.fetch_trajectories(output_file=args.data)
    return data_path.exists()


def run_training(args):
    algorithm = train.Algorithm(args.algorithm)
    model = train.train_model(
        data_path=args.data,
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
    logging.info('Running off-policy evaluation against %s', args.model_path)
    ope_module.evaluate_ope(trajectories_file=args.data, model_path=args.model_path, gamma=args.gamma)


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
