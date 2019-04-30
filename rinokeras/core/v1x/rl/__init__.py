from .StandardPolicy import StandardPolicy  # noqa: F401
from .RecurrentPolicy import RecurrentPolicy  # noqa: F401
from .LSTMPolicy import LSTMPolicy  # noqa: F401
from .RayPolicy import ray_policy
from .MultiModelRayPolicy import register_mm_ray_policy  # noqa: F401


def register_rinokeras_policies_with_ray():
    policies = [StandardPolicy, RecurrentPolicy, LSTMPolicy]
    for policy in policies:
        ray_policy(policy)
