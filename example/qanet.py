
import numpy as np
import tensorflow as tf


from dataflow.datasets.nlp.NewslensQA import NLQA  # Get the newslens QA dataset
from rl_algs.eager.models.qanet import QANet  # Get the QANet keras model


# Construct the dataset
dataset = NLQA()

# Construct the QANet encoder module
encoder_module = QANet(word_embed_matrix=, char_embed_matrix=)


## In progress