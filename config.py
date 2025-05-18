import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64
HIDDEN_DIM = 32
MAX_LENGTH = 30
INPUT_DIM = 28 # may change for language
OUTPUT_DIM = 65 # may change for language

TYPE = 'GRU'
ENCODER_NUM_LAYERS = 1
DECODER_NUM_LAYERS = 2
BATCH_FIRST = True
BATCH_SIZE = 128
DROPOUT_RATE = 0.1
BIDIRECTIONAL = True

PRINT_EVERY = 10
PLOT_EVERY = 100
LEARNING_RATE = 0.003