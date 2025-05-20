import torch
from config import *
from model import Encoder,BeamSearchDecoder,Seq2Seq_Model
from AttentionModel import Decoder_Attention,Attention_Network
from dataset import Dataset
import wandb
import argparse
import matplotlib
from utils import log_test_predictions_to_wandb,generate_word_heatmap


matplotlib.rcParams['font.family'] = ['Noto Sans', 'Noto Sans Devanagari', 'sans-serif']




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the Translation System.")

    parser.add_argument("-wp", "--wandb_project", type=str,help="Project name used to track experiments in Weights & Biases dashboard.",default='dl-assignment3')
    parser.add_argument("-we", "--wandb_entity", type=str,help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default='cs24s031')        
    parser.add_argument('-h_dim',"--hidden_dim",type=int,default=32,help='hidden dim size')
    parser.add_argument('-e_dim',"--embed_dim",type=int,default=64,help="Embedding layer dimension")    
    parser.add_argument('-e_layers',"--enc_layers",type=int,default=3,help="encoder layer number")
    parser.add_argument('-d_layers',"--dec_layers",type=int,default=3,help="deccoder layer number")    
    parser.add_argument('-c_type',"--cell_type",type=str,default="GRU",choices=['LSTM','GRU','RNN'],help='type of the encoder-decoder cell')
    parser.add_argument('-do',"--dropout",type=float,default=0.3,choices=[0.2,0.3,0.5,0.4,0.7,0.6],help='apply dropout')
        
    parser.add_argument("-b", "--batch_size", type=int, default=64,help="Batch size used to train the neural network.")
    parser.add_argument("-beam_size", "--beam_size", type=int, default=3,help="Beam search value that takes top k values.")
    parser.add_argument('-t_forcing',"--teacher_forcing",type=float,default=0.5,help="Teacher forcing ration used in decoder while training.")
    parser.add_argument('-logw',"--log_wandb",action='store_true',help="Enable Weights & Biases logging.")
    parser.add_argument("-arch","--architecture",type=str,default='attention',choices=['attention','vanilla'],help='Choose which architecture to use.')
    
    parser.add_argument('-lang',"--language",type=str,default='hi',choices=['hi','bn','gu','kn','ml','mr','pa','sd','ta','te','ur'],help='language you want to translate to.')
    parser.add_argument('-i_dim','--input_dim',type=int,default=28,help='Run the dataset.py seperately after making changes in the dataset.py, it will print the input dim, use that value.')
    parser.add_argument('-o_dim',"--output_dim",type=int,default=65,help='Run the dataset.py seperately after making changes in the dataset.py, it will print the output dim, use that value.')    

    args = parser.parse_args()
    batch_size = args.batch_size
    dropout = args.dropout
    log_wandb = args.log_wandb
    
    hidden_dim = args.hidden_dim
    embed_dim = args.embed_dim
    enc_layers = args.enc_layers
    dec_layers = args.dec_layers
    cell_type = args.cell_type
    teacher_forcing = args.teacher_forcing
    arch = args.architecture
    
    input_dim = args.input_dim
    output_dim = args.output_dim
    language = args.language
    beam_size = args.beam_size
    print(args)
    if log_wandb:
        wandb.login()
        wandb.init(project = args.wandb_project)
    
    input_lang,output_lang,train_loader,test_loader,valid_loader = Dataset().load_data(batch_size=batch_size,lang=language)
    if arch == 'vanilla':
        encoder = Encoder(cell_type=cell_type,num_layers=enc_layers,hidden_dim=hidden_dim,
                    embed_dim=embed_dim,input_dim=input_dim,dropout_rate=dropout,
                    bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST).to(DEVICE)
        decoder = BeamSearchDecoder(beam_size=beam_size,type=cell_type,num_layers=dec_layers,hidden_dim=hidden_dim,dropout_rate=dropout,
                          bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST,embed_dim=embed_dim,output_dim=output_dim)

        model = Seq2Seq_Model(encoder=encoder,decoder=decoder).to(DEVICE)
        state = torch.load('models/vanilla.pth',map_location = DEVICE)
    elif arch == 'attention':
        encoder = Encoder(cell_type=cell_type,num_layers=enc_layers,hidden_dim=hidden_dim,
                    embed_dim=embed_dim,input_dim=input_dim,dropout_rate=dropout,
                    bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST).to(DEVICE)
        decoder = Decoder_Attention(type=cell_type,num_layers=dec_layers,hidden_dim=hidden_dim,
                                           dropout_rate=dropout,bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST,
                                           embed_dim=embed_dim,output_dim=output_dim).to(DEVICE)
        model = Attention_Network(encoder=encoder,decoder=decoder).to(DEVICE)
        state = torch.load('models/attention_new.pth',map_location = DEVICE)
    else:
        print("Wrong architecture type")
        exit()
    model.load_state_dict(state)
    if log_wandb:
        log_test_predictions_to_wandb(model.encoder,model.decoder,test_loader=test_loader,input_lang=input_lang,output_lang=output_lang,name=arch)
    test_loss,test_acc = model.test_loss_acc(model.encoder,model.decoder,test_loader,teacher_ratio=teacher_forcing)
    print('Test Loss: {}, Test Acc: {}'.format(test_loss,test_acc))
    if model == 'attention':
        generate_word_heatmap(model.encoder,model.decoder,test_loader,input_lang,output_lang,
                          name='attention')