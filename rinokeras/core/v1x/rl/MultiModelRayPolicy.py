from typing import Dict, Type

import numpy as np
import tensorflow as tf
# import tensorflow.keras.backend as K
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.models.misc import linear, normc_initializer


def register_mm_ray_policy(name: str, policy_model: Type[tf.keras.Model], networks: Dict[str, tf.keras.Model]):
    """
    Constructs a Ray policy with multiple models as part of the collections. This
    allows for distributed training with multiple parameter sets (for example, 
    when using auxillary losses)
    
    Arguments:
        policy_model {Type[tf.keras.Model]} -- The policy model which is called to make predictions
        networks {Dict[str, tf.keras.Model]} -- A dictionary of additional networks
    
    Returns:
        ray.rllib.models.Model -- A model which can be used with Ray
    """

    class MMRayPolicy(ray.rllib.models.Model):

        @override(ray.rllib.models.Model)
        def _build_layers_v2(self, input_dict, num_outputs, options):

            # Setup the policy model
            if tf.get_collection('_rk_policy_model'):
                self.model = tf.get_collection('_rk_policy_model')[0]
            else:
                self.model = policy_model(num_outputs, **options)
                tf.add_to_collection('_rk_policy_model', self.model)

            # Add any other models to the collection
            if networks:
                self.networks = {}
            for key in networks.keys():
                if tf.get_collection('_rk_networks_{}'.format(key)):
                    self.networks[key] = tf.get_collection('_rk_networks_{}'.format(key))
                else:
                    self.networks[key] = [networks[key](**options), None, None]
                    tf.add_to_collection('_rk_networks_{}'.format(key), self.networks[key])

            if self.model.recurrent:
                self.state_init = [
                    np.zeros([state_size]) for state_size in self.model.state_size]

                if not self.state_in:
                    self.state_in = [tf.placeholder(tf.float32, [None, state_size])
                                     for state_size in self.model.state_size]

                output = self.model(input_dict,
                                    seqlens=self.seq_lens,
                                    initial_state=self.state_in)
                self.state_out = list(output['state_out'])
            else:
                output = self.model(input_dict)
            self.policy_output = output

            # Update the input dict with the model outputs
            input_dict['model_outputs'] = output

            # Compute the outputs for each of the networks
            for key, net in self.networks.items():
                if net[0].recurrent:
                    net[1] = [
                        np.zeros([state_size]) for state_size in net[0].state_size]

                    if not net[2]:
                        net[2] = [tf.placeholder(tf.float32, [None, state_size])
                                        for state_size in net[0].state_size]

                    self.network_outputs[key] = net[0](input_dict,
                                                        seqlens=self.seq_lens,
                                                        initial_state=net[2])
                else:
                    self.network_outputs[key] = net[0](input_dict)

            return output['logits'], output['latent']

        @override(ray.rllib.models.Model)
        def custom_loss(self, policy_loss, loss_inputs):


            # Update the loss_inputs with all of the model outputs
            if self.networks:
                loss_inputs['network_outputs'] = {k:self.network_outputs[k] for k in self.networks.keys()}
                loss_inputs['network_outputs']['policy_model'] = self.policy_output

            total_loss = policy_loss
            if hasattr(self.model, 'custom_loss'):
                total_loss = self.model.custom_loss(policy_loss, loss_inputs)

            if self.networks:
                for _, net in self.networks.items():
                    if hasattr(net[0], 'custom_loss'):
                        total_loss = net[0].custom_loss(total_loss, loss_inputs, )
                return total_loss

    MMRayPolicy.__name__ = name
    MMRayPolicy.__doc__ = "Wraped Multi-Network RAY policy"

    ModelCatalog.register_custom_model(name, MMRayPolicy)

    return MMRayPolicy
