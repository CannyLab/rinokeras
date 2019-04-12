"""Example of a custom gym environment and model. Run this for a demo.

This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search

You can visualize experiment results in ~/ray_results using TensorBoard.
"""
from typing import Optional, Sequence
import yaml

from rinokeras.rl import register_rinokeras_policies_with_ray, ray_policy, StandardPolicy

import ray
from ray import tune


if __name__ == "__main__":
    register_rinokeras_policies_with_ray()
    ray.init()
    with open('atari-ppo.yaml') as f:
        config = yaml.load(f)
    tune.run(**config)
