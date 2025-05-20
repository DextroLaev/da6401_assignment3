"""
run_sweep.py

Run a Weights & Biases hyperparameter sweep for a vanilla encoder–decoder
sequence-to-sequence translation model with optional beam search or Gumbel decoding.

This script defines a `sweep_train` function that:
  - Logs in to W&B and initializes a new run with hyperparameters from the sweep config.
  - Loads the dataset and builds the encoder and decoder models based on the sweep parameters.
  - Trains the Seq2Seq model, logging training and validation metrics (and optionally test evaluation) to W&B.

When executed as the main module, it:
  - Defines a Bayesian sweep configuration over embedding size, number of layers,
    hidden dimensions, RNN cell type, dropout rate, learning rate, teacher forcing ratio,
    batch size, and beam size.
  - Launches the W&B agent to run the specified number of sweep trials.
"""

import wandb
from config import *
from dataset import Dataset

from model import Encoder,BeamSearchDecoder,Seq2Seq_Model

def sweep_train():
    """
    Perform one training run for the current hyperparameter configuration in the W&B sweep.

    This function:
      1. Logs in to Weights & Biases and initializes a run named according to the model
         configuration (cell type, embedding size, hidden dimension, encoder/decoder layers).
      2. Loads the train, validation, and test datasets for the specified language.
      3. Constructs the encoder and decoder models using the sweep parameters:
         - Embedding dimension, hidden dimension
         - Number of encoder and decoder layers
         - RNN cell type (RNN/GRU/LSTM)
         - Dropout rate
         - Beam size (with optional Gumbel decoding if beam_size == 1)
      4. Wraps them in a Seq2Seq_Model and trains for a fixed number of epochs,
         logging metrics (training/validation loss and accuracy, plus test evaluation)
         back to W&B.
    """

    wandb.login()
    var1 = wandb.init(project='dl-assignment3')
    w_config = var1.config
    batch_size = w_config.batch_size
    cell_type = w_config.cell_type
    enc_layers = w_config.enc_layers
    dec_layers = w_config.dec_layers
    hidden_dim = w_config.hidden_dim
    dropout = w_config.dropout
    embed_dim = w_config.embed_size
    lr = w_config.lr
    batch_size = w_config.batch_size
    beam_size = w_config.beam_size
    teacher_forcing = w_config.teacher_forcing

    run_name = f"cell_{cell_type}_embedDim_{embed_dim}_hiddenDim_{hidden_dim}_encLayer_{enc_layers}_decLayers_{dec_layers}"
    
    wandb.run.name = run_name
    wandb.run.save()

    input_lang,output_lang,train_loader,test_loader,valid_loader = Dataset().load_data(batch_size=batch_size,lang='hi')
    print("Data Loading is complete..")
    encoder = Encoder(cell_type=cell_type,num_layers=enc_layers,hidden_dim=hidden_dim,
                    embed_dim=embed_dim,input_dim=INPUT_DIM,dropout_rate=dropout,
                    bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST).to(DEVICE)
    
    decoder = BeamSearchDecoder(type=cell_type,num_layers=dec_layers,hidden_dim=hidden_dim,dropout_rate=dropout,
                  bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST,embed_dim=embed_dim,output_dim=OUTPUT_DIM
                  ,beam_size=beam_size, use_gumbel=True if beam_size == 1 else False, temperature=0.8).to(DEVICE)
    
    print('Encoder-Decoder model Initiated')
    seq2seq = Seq2Seq_Model(encoder,decoder).to(DEVICE)

    print('Sequence to Sequence model initiated')
    print('Starting Model Training..')
    seq2seq.train_model(train_loader,valid_loader,input_lang,output_lang,test_loader=test_loader,epochs=20,wandb_log=True,learning_rate=lr,teacher_ratio=teacher_forcing,evaluate_test=True) 
    

if __name__ == '__main__':
   
    sweep_config = {
		'name': 'seq2seq-exp(bayes-select)-LAB',
		'method': 'bayes',
		'metric': {'goal': 'maximize', 'name': 'val_acc'},
		'parameters': {
		    'embed_size':{'values':[16,32,64,256]},		    
		    'enc_layers': {'values': [1,2,3]},
		    'dec_layers': {'values': [1,2,3]},
		    'hidden_dim': {'values': [16,32,64,256]},
		    'cell_type': {'values': ['RNN','GRU','LSTM']},
            'dropout':{'values':[0.2,0.3]},        
            'lr':{'values':[0.1,0.01,0.001]},
            'teacher_forcing':{'values':[0.2,0.3,0.5,0.6]},
            'batch_size':{'values':[16,32,64]},
            'beam_size':{'values':[1,2,3]}
		  }
    }
    sweep_id = wandb.sweep(sweep_config,project='dl-assignment3')
    wandb.agent(sweep_id,sweep_train,count=50)
    wandb.finish()
