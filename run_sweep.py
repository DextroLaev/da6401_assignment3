import wandb
from config import *
from dataset import Dataset

from model import Encoder,BeamSearchDecoder,Seq2Seq_Model

def sweep_train():
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
    epochs = w_config.epochs
    lr = w_config.lr
    batch_size = w_config.batch_size
    beam_size = w_config.beam_size

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
    seq2seq.train_model(train_loader,valid_loader,epochs=epochs,wandb_log=True,learning_rate=lr,teacher_ratio=0.5) 
    

if __name__ == '__main__':
   
    sweep_config = {
		'name': 'seq2seq-exp(bayes-select)',
		'method': 'bayes',
		'metric': {'goal': 'maximize', 'name': 'val_acc'},
		'parameters': {
		    'embed_size':{'values':[16,32,64,256]},		    
		    'enc_layers': {'values': [1,2,3]},
		    'dec_layers': {'values': [1,2,3]},
		    'hidden_dim': {'values': [16,32,64,256]},
		    'cell_type': {'values': ['RNN','GRU','LSTM']},
        'dropout':{'values':[0.2,0.3]},
        'epochs':{'values':[10,20,30]},
        'lr':{'values':[0.1,0.01,0.001]},
        'teacher_forcing':{'values':[0.2,0.3,0.5,0.6]},
        'batch_size':{'values':[16,32,64]},
        'beam_size':{'values':[1,2,3]}
		  }
    }
    sweep_id = wandb.sweep(sweep_config,project='dl-assignment3')
    wandb.agent(sweep_id,sweep_train,count=50)
    wandb.finish()
