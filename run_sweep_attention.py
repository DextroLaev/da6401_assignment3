import wandb
from config import *
from dataset import Dataset

from model import Encoder
from AttentionModel import Attention_Network,Decoder_Attention

def sweep_train():
    print('trying to login')
    wandb.login(key='702a37b01ca39351ffb1a500f0354b1c31a63920')
    print('Done logging, creating sweep')
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
                           wandb_log=True,learning_rate=lr,teacher_ratio=0.5,evaluate_test=True,heatmap=False)
    

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
    wandb.agent(sweep_id,sweep_train,count=80)
    wandb.finish()