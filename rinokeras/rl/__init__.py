from rinokeras import RK_USE_TF_VERSION as _RK_USE_TF_VERSION

if _RK_USE_TF_VERSION == 1:
    from rinokeras.core.v1x.rl import StandardPolicy
    from rinokeras.core.v1x.rl import RecurrentPolicy
    from rinokeras.core.v1x.rl import LSTMPolicy
    from rinokeras.core.v1x.rl import ray_policy
    from rinokeras.core.v1x.rl import register_rinokeras_policies_with_ray
elif _RK_USE_TF_VERSION == 2:
    pass
