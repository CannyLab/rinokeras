from typing import Type

import numpy as np
import tensorflow as tf
# import tensorflow.keras.backend as K
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.models.misc import linear, normc_initializer


def ray_policy(model: Type[tf.keras.Model]):
    "Wraps a keras model and turns it into a Ray Model"

    class WrappedRayPolicy(ray.rllib.models.Model):

        @override(ray.rllib.models.Model)
        def _build_layers_v2(self, input_dict, num_outputs, options):
            self.obs_in = input_dict["obs"]
            if tf.get_collection('model'):
                self.model = tf.get_collection('model')[0]
            else:
                self.model = model(num_outputs, **options)
                tf.add_to_collection('model', self.model)

            if self.model.recurrent:
                dummy = tf.placeholder(tf.float32, [None])
                initial_state = self.model.get_initial_state(dummy)

                if not self.state_in:
                    self.state_in = \
                        [tf.placeholder(state.dtype, state.shape) for state in initial_state]

                def get_numpy_state(state):
                    shape = state.shape.as_list()
                    if shape[0] is None:
                        shape = shape[1:]
                    assert not any(s is None for s in shape), 'Found None in initial state shape'
                    dtype = state.dtype.as_numpy_dtype
                    return np.zeros(shape, dtype)

                self.state_init = [get_numpy_state(state) for state in initial_state]

                output = self.model(input_dict, seqlens=self.seq_lens, initial_state=self.state_in)
                self.state_out = output['state_out']
            else:
                output = self.model(input_dict)

            return output['logits'], output['latent']

        @override(ray.rllib.models.Model)
        def custom_loss(self, policy_loss, loss_inputs):
            if hasattr(self.model, 'custom_loss'):
                return self.model.custom_loss(policy_loss, loss_inputs)
            return policy_loss

        @override(ray.rllib.models.Model)
        def custom_stats(self,):
            if hasattr(self.model, 'custom_stats'):
                return self.model.custom_stats()
            return {}

    WrappedRayPolicy.__name__ = model.__name__
    WrappedRayPolicy.__doc__ = model.__doc__

    ModelCatalog.register_custom_model(model.__name__, WrappedRayPolicy)

    return WrappedRayPolicy
