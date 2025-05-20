"""
run_sweep_attention.py

Run a Weights & Biases hyperparameter sweep for an attention-based
sequence-to-sequence translation model.

This script defines a `sweep_train` function that:
  1. Logs in to W&B with an API key.
  2. Initializes a new run named according to the current hyperparameter configuration.
  3. Loads the dataset and builds an encoder and attention decoder based on the sweep parameters.
  4. Trains the Attention_Network model, logging metrics (training/validation loss & accuracy,
     plus optional test evaluation) to W&B.

When executed as the main module, it:
  - Defines a Bayesian sweep configuration over embedding size, number of layers,
    hidden dimensions, RNN cell type, dropout rate, learning rate, teacher forcing ratio,
    and batch size.
  - Launches the W&B agent to run the specified number of sweep trials.
"""

import wandb
from config import *
from dataset import Dataset

from model import Encoder
from AttentionModel import Attention_Network,Decoder_Attention

def sweep_train():
    """
    Execute one training run for the current hyperparameter configuration in the W&B sweep.

    This function performs the following steps:
      1. Logs in to Weights & Biases .
      2. Initializes a new W&B run and retrieves the hyperparameters from the sweep config.
      3. Names the run based on cell type, embedding size, hidden dimension,
         and number of encoder/decoder layers.
      4. Loads the training, validation, and test datasets for the Hindi language.
      5. Constructs:
         - An Encoder with the specified cell type, number of layers,
           hidden dimension, embedding dimension, dropout rate, and device settings.
         - A Decoder_Attention with the same hyperparameters for attention.
      6. Wraps them in an Attention_Network and trains for a fixed number of epochs,
         logging metrics back to W&B.
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
    teacher_ratio = w_config.teacher_forcing

    run_name = f"attention_cell_{cell_type}_embedDim_{embed_dim}_hiddenDim_{hidden_dim}_encLayer_{enc_layers}_decLayers_{dec_layers}"
    
    wandb.run.name = run_name
    wandb.run.save()

    input_lang,output_lang,train_loader,test_loader,valid_loader = Dataset().load_data(batch_size=batch_size,lang='hi')
    print("Data Loading is complete..")
    encoder = Encoder(cell_type=cell_type,num_layers=enc_layers,hidden_dim=hidden_dim,
                    embed_dim=embed_dim,input_dim=INPUT_DIM,dropout_rate=dropout,
                    bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST).to(DEVICE)
    
    attention_decoder = Decoder_Attention(type=cell_type,num_layers=dec_layers,hidden_dim=hidden_dim,
                                          dropout_rate=dropout,bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST,
                                          embed_dim=embed_dim,output_dim=OUTPUT_DIM).to(DEVICE)
    
    print('Attention Encoder-Decoder model Initiated')
    attn_model = Attention_Network(encoder=encoder,decoder=attention_decoder)

    print('Sequence to Sequence model initiated')
    print('Starting Model Training..')
    attn_model.train_model(train_loader,valid_loader,input_lang,output_lang,test_loader,epochs=20,
                           wandb_log=True,learning_rate=lr,teacher_ratio=teacher_ratio,evaluate_test=True,heatmap=False)
    

if __name__ == '__main__':
   
    sweep_config = {
		'name': 'seq2seq-exp(bayes-select)-LAB(Attention)',
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
		  }
    }
    sweep_id = wandb.sweep(sweep_config,project='dl-assignment3')
    wandb.agent(sweep_id,sweep_train,count=50)
    wandb.finish()