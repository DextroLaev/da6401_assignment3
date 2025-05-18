import torch
from config import *
from model import Encoder,Decoder,Seq2Seq_Model
from AttentionModel import Decoder_Attention,Attention_Network
from dataset import Dataset

if __name__ == '__main__':
    input_lang,output_lang,train_loader,test_loader,valid_loader = Dataset().load_data(batch_size=16,lang='hi')
    
    
    encoder = Encoder(cell_type='LSTM',num_layers=3,hidden_dim=256,
                embed_dim=64,input_dim=28,dropout_rate=0.2,
                bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST).to(DEVICE)
    decoder = Decoder_Attention(type='LSTM',num_layers=3,hidden_dim=256,
                                        dropout_rate=0.2,bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST,
                                        embed_dim=64,output_dim=65).to(DEVICE)
    model = Attention_Network(encoder=encoder,decoder=decoder).to(DEVICE)

    state = torch.load('models/attention_state.pth',map_location = DEVICE)
    model.load_state_dict(state)
    print(model.test_loss_acc(model.encoder,model.decoder,test_loader,teacher_ratio=0.5))
    