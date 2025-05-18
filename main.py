from dataset import Dataset
from config import *
import argparse
import wandb
from model import Encoder,Seq2Seq_Model,BeamSearchDecoder
from AttentionModel import Attention_Network,Decoder_Attention

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Translation System.")

    parser.add_argument("-wp", "--wandb_project", type=str,help="Project name used to track experiments in Weights & Biases dashboard.",default='dl-assignment3')
    parser.add_argument("-we", "--wandb_entity", type=str,help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default='cs24s031')        
    parser.add_argument('-h_dim',"--hidden_dim",type=int,default=32,help='hidden dim size')
    parser.add_argument('-e_dim',"--embed_dim",type=int,default=64,help="Embedding layer dimension")    
    parser.add_argument('-e_layers',"--enc_layers",type=int,default=3,help="encoder layer number")
    parser.add_argument('-d_layers',"--dec_layers",type=int,default=3,help="deccoder layer number")    
    parser.add_argument('-c_type',"--cell_type",type=str,default="GRU",choices=['LSTM','GRU','RNN'],help='type of the encoder-decoder cell')
    parser.add_argument('-do',"--dropout",type=float,default=0.3,choices=[0.2,0.3,0.5,0.4,0.7,0.6],help='apply dropout')
    
    parser.add_argument("-e", "--epochs", type=int, default=25,help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,help="Batch size used to train the neural network.")
    
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,help="Learning rate used to optimize model parameters.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.00005,help="Weight decay used by optimizers.")
    parser.add_argument("-beam_size", "--beam_size", type=int, default=3,help="Beam search value that takes top k values.")
    parser.add_argument('-t_forcing',"--teacher_forcing",type=float,default=0.5,help="Teacher forcing ration used in decoder while training.")
    parser.add_argument('--save_model',type=bool,default=False,choices=[True,False],help="If you want to save the trained model, set it to True")
    parser.add_argument('-logw',"--log_wandb",type=bool,default=False,choices=[True,False],help="If you want to log the performance of the model, set it to True")
    parser.add_argument("-arch","--architecture",type=str,default='transformer',choices=['transformer','vanilla'],help='Choose which architecture to use.')
    parser.add_argument("--evaluate",type=bool,default=False,choices=[True,False],help='Make it True if you want to get testing accuracy.')
    parser.add_argument('-lang',"--language",type=str,default='hi',choices=['hi','bn','gu','kn','ml','mr','pa','sd','ta','te','ur'],help='language you want to translate to.')
    parser.add_argument('-i_dim','--input_dim',type=int,default=28,help='Run the dataset.py seperately after making changes in the dataset.py, it will print the input dim, use that value.')
    parser.add_argument('-o_dim',"--output_dim",type=int,default=65,help='Run the dataset.py seperately after making changes in the dataset.py, it will print the output dim, use that value.')    

    args = parser.parse_args()
    
    print(args)
    lr = args.learning_rate
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    dropout = args.dropout
    save_model = args.save_model
    log_wandb = args.log_wandb
    epochs = args.epochs
    hidden_dim = args.hidden_dim
    embed_dim = args.embed_dim
    enc_layers = args.enc_layers
    dec_layers = args.dec_layers
    cell_type = args.cell_type
    teacher_forcing = args.teacher_forcing
    arch = args.architecture
    evaluate = args.evaluate
    input_dim = args.input_dim
    output_dim = args.output_dim
    language = args.language
    beam_size = args.beam_size
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
    elif arch == 'transformer':
        encoder = Encoder(cell_type=cell_type,num_layers=enc_layers,hidden_dim=hidden_dim,
                    embed_dim=embed_dim,input_dim=input_dim,dropout_rate=dropout,
                    bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST).to(DEVICE)
        decoder = Decoder_Attention(type=cell_type,num_layers=dec_layers,hidden_dim=hidden_dim,
                                           dropout_rate=dropout,bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST,
                                           embed_dim=embed_dim,output_dim=output_dim).to(DEVICE)
        model = Attention_Network(encoder=encoder,decoder=decoder).to(DEVICE)
    else:
        print("Wrong architecture type")
        exit()
    
    model.train_model(train_loader=train_loader,valid_loader=valid_loader,test_loader=test_loader,input_lang=input_lang,
                      output_lang=output_lang,epochs=epochs,wandb_log=log_wandb,learning_rate=lr,
                      teacher_ratio=teacher_forcing,evaluate_test=evaluate,save_model=save_model
                      )