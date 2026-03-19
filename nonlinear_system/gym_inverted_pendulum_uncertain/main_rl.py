#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training RL algorithms on pendulum environment.
"""

import argparse
import os
import torch
import traceback
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback

# Import functions from rl_baseline
from training.rl_baseline import evaluate_rl_controller, test_model, plot_actions
from uncertain_env import UncertainDisturbancePendulumEnv


def ensure_directory(directory):
    """Ensure a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RL Training for Pendulum Control")
    parser.add_argument(
        "--algo", type=str, default="sac", choices=["ppo", "sac"], help="RL algorithm to use"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="nonlinear_system/gym_inverted_pendulum_uncertain/saved_models",
        help="Directory to save models",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate and visualize after training"
    )
    parser.add_argument(
        "--test_steps", type=int, default=300, help="Number of steps to test the model"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500000, help="Number of timesteps to train"
    )
    parser.add_argument(
        "--disturbance_max",
        type=float,
        default=0.5,
        help="Bound for uncertain disturbance torque sampled in [-d, d]",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5000,
        help="Print training progress every N environment steps",
    )
    parser.add_argument(
        "--rollout_eval_steps",
        type=int,
        default=200,
        help="Number of steps for periodic deterministic rollout evaluation",
    )
    return parser.parse_args()


class TrainingProgressCallback(BaseCallback):
    """Print periodic training progress and short rollout performance."""

    def __init__(self, eval_env, log_interval=5000, rollout_eval_steps=200, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_interval = int(log_interval)
        self.rollout_eval_steps = int(rollout_eval_steps)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval != 0:
            return True

        mean_ep_rew = None
        mean_ep_len = None
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            mean_ep_rew = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            mean_ep_len = np.mean([ep["l"] for ep in self.model.ep_info_buffer])

        obs, _ = self.eval_env.reset()
        rollout_return = 0.0
        for _ in range(self.rollout_eval_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.eval_env.step(action)
            rollout_return += float(reward)
            if terminated or truncated:
                obs, _ = self.eval_env.reset()

        if mean_ep_rew is None:
            print(
                f"[train] steps={self.num_timesteps} rollout_return({self.rollout_eval_steps})={rollout_return:.3f}"
            )
        else:
            print(
                f"[train] steps={self.num_timesteps} "
                f"mean_ep_rew={mean_ep_rew:.3f} mean_ep_len={mean_ep_len:.1f} "
                f"rollout_return({self.rollout_eval_steps})={rollout_return:.3f}"
            )
        return True


def main():
    """Main function for RL training."""
    args = parse_args()

    # Ensure directories exist
    ensure_directory(args.save_dir)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set up uncertain environment
    test_env = UncertainDisturbancePendulumEnv(
        render_mode="human", g=9.81, m=1.0, l=1.0, b=0.13, disturbance_max=args.disturbance_max
    )
    eval_env = UncertainDisturbancePendulumEnv(
        g=9.81, m=1.0, l=1.0, b=0.13, disturbance_max=args.disturbance_max
    )
    print(f"Using uncertain pendulum environment (disturbance_max={args.disturbance_max})")

    # ----- Determine model paths and see if we need to load previous checkpoint-----------------------------------------------------------------
    standard_model_path = os.path.join(args.save_dir, args.algo, "pendulum_uncertain.zip")
    model_path = os.path.join(args.save_dir, f"{args.algo}_pendulum_uncertain")
    disturbance_tag = str(args.disturbance_max).replace(".", "p")
    env_suffix = f"_uncertain_d{disturbance_tag}"
    model_path += env_suffix

    model = None
    load_path = None

    # Check if model exists in either location
    if os.path.exists(standard_model_path):
        load_path = standard_model_path.replace(".zip", "")
    elif os.path.exists(f"{model_path}.zip"):
        load_path = model_path

    # Load existing model if available, otherwise train
    if load_path is not None:
        # Load existing model
        print(f"Loading existing model from: {load_path}.zip")
        if args.algo == "ppo":
            model = PPO.load(load_path)
        else:  # SAC
            model = SAC.load(load_path)
        
        if not args.evaluate:
            print("Model loaded. Use --evaluate flag to evaluate it.")
    else:
        # No existing model: train a new one
        if args.evaluate:
            print("Error: No trained model found for evaluation.")
            print(f"  Checked: {standard_model_path}")
            print(f"  Checked: {model_path}.zip")
            print("Please train a model first (run without --evaluate flag).")
            return
        
        print(f"Training {args.algo} on pendulum with uncertain disturbance...")
        train_env = UncertainDisturbancePendulumEnv(
            g=9.81, m=1.0, l=1.0, b=0.13, disturbance_max=args.disturbance_max
        )

        # Create and train model
        if args.algo == "ppo":
            model = PPO("MlpPolicy", train_env, verbose=1)
        else:  # SAC
            model = SAC("MlpPolicy", train_env, verbose=1)

        progress_callback = TrainingProgressCallback(
            eval_env=eval_env,
            log_interval=args.log_interval,
            rollout_eval_steps=args.rollout_eval_steps,
        )
        model.learn(total_timesteps=args.timesteps, callback=progress_callback)
        model.save(model_path)
        print(f"Model saved to {model_path}.zip")

    if args.evaluate and model is not None:
        try:

            print("Evaluating model:", model_path)

            # Generate value function and policy plots
            try:
                evaluate_rl_controller(model, env=eval_env)
            except Exception as e:
                print(f"Warning: Could not run visual evaluation: {e}")
                traceback.print_exc()

            # Test the model
            print("Testing model...")
            try:
                actions, values = test_model(model, test_env, num_steps=args.test_steps)
                plot_actions(
                    values,
                    values,
                    f"{args.algo.upper()} Value Function Over Time ({env_suffix})",
                    os.path.join(args.save_dir, f"value_function_{args.algo}{env_suffix}.png"),
                    y_axis_name="Value",
                )
                plot_actions(
                    actions,
                    actions,
                    f"{args.algo.upper()} Actions Over Time ({env_suffix})",
                    os.path.join(args.save_dir, f"actions_{args.algo}{env_suffix}.png"),
                    y_axis_name="Control Input",
                )
            except Exception as e:
                print(f"Warning: Could not run testing: {e}")
                traceback.print_exc()

        except Exception as e:
            print(f"Error loading or evaluating model: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
