import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
EMBED_DIM = 64
HIDDEN_DIM = 256
MAX_LENGTH = 30
INPUT_DIM = 28 # may change for language
OUTPUT_DIM = 65 # may change for language

TYPE = 'LSTM'
ENCODER_NUM_LAYERS = 3
DECODER_NUM_LAYERS = 3
BATCH_FIRST = True
BATCH_SIZE = 16
DROPOUT_RATE = 0.3
BIDIRECTIONAL = True
EPOCHS = 30

LEARNING_RATE = 0.001
TEACHER_FORCING_VANILLA = 0.5
TEACHER_FORCING_ATTENTION = 0.5

WANDB_LOG = True
SAVE_MODEL = True