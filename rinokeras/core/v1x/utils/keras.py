
import tensorflow.keras.backend as K

def load_distributed(distribution_strategy, model, filename, by_name=False):
    with distribution_strategy.scope():
        model.load_weights(filename, by_name=by_name)
        weights = model.get_weights()
        assign_ops = []

        for layer in model.layers:
            num_param = len(layer.weights)
            layer_weights = weights[:num_param]

            for sw, w in zip(layer.weights, layer_weights):
                assign_ops.append(distribution_strategy.unwrap(sw.assign(w)))
            weights = weights[num_param:]
        K.get_session().run(assign_ops)
