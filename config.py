
import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf
# import torch
tf.compat.v1.enable_eager_execution()

gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    tf.config.set_visible_devices(gpu_devices[0], 'GPU')

# DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 10
LATENT_DIM = 256
BATCH_SIZE = 64
