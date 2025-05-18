import torch
from config import *
from dataset import Dataset
from model import Seq2Seq_Model,Encoder,BeamSearchDecoder
from AttentionModel import Attention_Network,Decoder_Attention

if __name__ == '__main__':
    input_lang,output_lang,train_loader,test_loader,valid_loader = Dataset().load_data(batch_size=32,lang='hi')
    print("Data Loading is complete..")
    encoder = Encoder(cell_type=TYPE,num_layers=ENCODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,
                    embed_dim=EMBED_DIM,input_dim=INPUT_DIM,dropout_rate=DROPOUT_RATE,
                    bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST).to(DEVICE)
    beam_size = [1, 2, 4]
    for i in beam_size:
        decoder = BeamSearchDecoder(type=TYPE,num_layers=DECODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,dropout_rate=DROPOUT_RATE,
                      bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST,embed_dim=EMBED_DIM,output_dim=OUTPUT_DIM
                      ,beam_size=i, use_gumbel=True if i == 1 else False, temperature=0.8).to(DEVICE)
        
        print('Encoder-Decoder model Initiated')
        seq2seq = Seq2Seq_Model(encoder,decoder).to(DEVICE)

        print('Sequence to Sequence model initiated')
        print('Starting Model Training..')
        seq2seq.train_model(train_loader,valid_loader,input_lang,output_lang,epochs=30,wandb_log=False,learning_rate=LEARNING_RATE,teacher_ratio=0.5,evaluate_test=False)    
    # attention_decoder = Decoder_Attention(type=TYPE,num_layers=DECODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,
    #                                       dropout_rate=DROPOUT_RATE,bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST,
    #                                       embed_dim=EMBED_DIM,output_dim=OUTPUT_DIM).to(DEVICE)
    # attn_model = Attention_Network(encoder=encoder,decoder=attention_decoder)
    # print('Starting model training...')
    # attn_model.train_model(train_loader,valid_loader,input_lang,output_lang,test_loader,epochs=30,
    #                        wandb_log=False,learning_rate=0.001,teacher_ratio=0.5,evaluate_test=True,heatmap=False)