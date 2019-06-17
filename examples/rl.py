"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
from typing import Optional, Sequence
import yaml

from rinokeras.rl import register_rinokeras_policies_with_ray, ray_policy

import ray
from ray.rllib.models import ModelCatalog
from ray import tune


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run rinokeras models with Ray')
    parser.add_argument('policy', type=str, default='StandardPolicy')
    args = parser.parse_args()

    register_rinokeras_policies_with_ray()
    ray.init()
    with open('atari-ppo.yaml') as f:
        config = yaml.load(f)
    config['config']['model']['custom_model'] = args.policy
    tune.run(**config)
