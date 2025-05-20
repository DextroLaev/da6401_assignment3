"""
train.py

Script to train and evaluate a character-level seq2seq translation model.
Supports two architectures: vanilla (beam search decoder) and attention-based.
Logs metrics and visualizations to Weights & Biases (WandB).
"""
import argparse
import torch
from config import *
from dataset import Dataset
from model import Seq2Seq_Model, Encoder, BeamSearchDecoder
from AttentionModel import Attention_Network, Decoder_Attention


def main():
    """
    Parse command-line arguments, load data, initialize models, train, and evaluate.

    This function:
      1. Parses the --architecture argument to choose between vanilla and attention models.
      2. Loads Hindi translation datasets.
      3. Instantiates encoder and appropriate decoder.
      4. Trains the model with default hyperparameters and logs to WandB.
      5. Evaluates on the test set and prints accuracy and loss.

    Raises:
      SystemExit: If an invalid architecture is specified.
    """
    parser = argparse.ArgumentParser(description="Train a Translation System with default settings.")
    parser.add_argument("-arch", "--architecture",default='attention', type=str,choices=['attention', 'vanilla'],help="Which encoder-decoder model to train: 'attention' or 'vanilla'.")

    args = parser.parse_args()
    arch = args.architecture

    # Load data
    input_lang, output_lang, train_loader, test_loader, valid_loader = Dataset().load_data(batch_size=BATCH_SIZE, lang='hi')
    print("Data loading complete.")

    # Initialize encoder
    encoder = Encoder(cell_type=TYPE,num_layers=ENCODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,embed_dim=EMBED_DIM,input_dim=INPUT_DIM,dropout_rate=DROPOUT_RATE,
                      bidirectional=BIDIRECTIONAL,
                      batch_first=BATCH_FIRST).to(DEVICE)

    if arch == 'vanilla':
        # Vanilla model with beam search decoder
        decoder = BeamSearchDecoder(type=TYPE,num_layers=DECODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,dropout_rate=DROPOUT_RATE,bidirectional=BIDIRECTIONAL,
                                    batch_first=BATCH_FIRST,embed_dim=EMBED_DIM,output_dim=OUTPUT_DIM,beam_size=3,temperature=0.8).to(DEVICE)
        
        model = Seq2Seq_Model(encoder, decoder).to(DEVICE)
        model.train_model(train_loader,valid_loader,input_lang,output_lang,epochs=EPOCHS,wandb_log=WANDB_LOG,learning_rate=LEARNING_RATE,
                        teacher_ratio=TEACHER_FORCING_VANILLA,save_model=SAVE_MODEL)
        
        model.test_loss_acc(encoder_model=encoder,decoder_model=decoder,dataloader=test_loader,teacher_ratio=0.0)

    elif arch == 'attention':
        # Attention-based model
        attention_decoder = Decoder_Attention(type=TYPE,num_layers=DECODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,dropout_rate=DROPOUT_RATE,bidirectional=BIDIRECTIONAL,
                                            batch_first=BATCH_FIRST,
                                            embed_dim=EMBED_DIM,
                                            output_dim=OUTPUT_DIM).to(DEVICE)
        model = Attention_Network(encoder=encoder,decoder=attention_decoder)
        model.train_model(train_loader,valid_loader,input_lang,output_lang,test_loader,epochs=EPOCHS,wandb_log=WANDB_LOG,learning_rate=LEARNING_RATE,
                        teacher_ratio=TEACHER_FORCING_ATTENTION,heatmap=False,save_model=SAVE_MODEL)
        
        model.test_loss_acc(encoder_model=encoder,decoder_model=attention_decoder,dataloader=test_loader,teacher_ratio=0.0)

    else:
        # Should never happen due to argparse choices
        print('Invalid architecture type chosen.')
        exit(1)


if __name__ == '__main__':
    main()
