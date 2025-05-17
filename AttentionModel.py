from model import Encoder,cell
import torch
from config import *
from dataset import start,end
import random
import numpy as np
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


class Attention_Module(torch.nn.Module):
    def __init__(self,layers,hidden_dim, bidirectional= False):
        super(Attention_Module,self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.layers = layers
        
        self.W_attn = torch.nn.Linear(hidden_dim*(1+bidirectional),hidden_dim*(1+bidirectional),bias=False)
        self.U_attn = torch.nn.Linear(hidden_dim*(1+bidirectional),hidden_dim*(1+bidirectional),bias=False)
        self.V_attn = torch.nn.Linear(hidden_dim*(1+bidirectional),1,bias=False)        

    def forward(self, query,keys):
        if self.bidirectional:
            query_t = torch.cat([query[:,-2,:],query[:,-1,:]],dim=1).unsqueeze(1)
            scores = self.V_attn(torch.tanh(self.W_attn(query_t)) + self.U_attn(keys))
        else:
            scores = self.V_attn(torch.tanh(self.W_attn(query[:,-1,:].unsqueeze(1))) + self.U_attn(keys))
        scores = scores.squeeze(2).unsqueeze(1)
        weight = torch.nn.functional.softmax(scores,dim=-1)
        context = torch.bmm(weight,keys)
        return context,weight            

class Decoder_Attention(torch.nn.Module):
    def __init__(self,type= TYPE,num_layers=DECODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,dropout_rate=DROPOUT_RATE,bidirectional=BIDIRECTIONAL,
                 batch_first=BATCH_FIRST,embed_dim=EMBED_DIM,output_dim=OUTPUT_DIM):
        super(Decoder_Attention, self).__init__()
        self.type = type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.dropout_rate = 0 if num_layers <= 1 else dropout_rate
        self.bidirectional = bidirectional
        
        self.attention_module = Attention_Module(self.num_layers,self.hidden_dim,self.bidirectional)
        self.embedding = torch.nn.Embedding(self.output_dim, self.embed_dim)

        rnn_input_data = self.embed_dim + self.hidden_dim*(1+bidirectional)
        self.cell = cell(type)(rnn_input_data, self.hidden_dim, num_layers=num_layers, batch_first=batch_first, dropout=self.dropout_rate, bidirectional=bidirectional)
        self.out = torch.nn.Linear(self.hidden_dim*(1+self.bidirectional), self.output_dim)
        self.dropout = torch.nn.Dropout(self.dropout_rate)

    def forward(self,encoder_outputs,encoder_hidden,target_tensor=None,teacher_ratio= 0.5,):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size,1,dtype=torch.long,device=DEVICE).fill_(start)
        decoder_outputs = []
        attentions = []
        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            if encoder_hidden.shape[0] != self.num_layers*(1+self.bidirectional):
                encoder_hidden_reshaped = torch.stack([encoder_hidden.mean(0) for i in range(self.num_layers*(1+self.bidirectional))])
                encoder_cell_reshaped = torch.stack([encoder_cell.mean(0) for i in range(self.num_layers*(1+self.bidirectional))])
                decoder_cell = encoder_cell_reshaped
                decoder_hidden = encoder_hidden_reshaped
            else:
                decoder_cell = encoder_cell
                decoder_hidden = encoder_hidden
        else:
            if encoder_hidden.shape[0] != self.num_layers*(1+self.bidirectional):
                encoder_hidden_reshaped = torch.stack([encoder_hidden.mean(0) for i in range(self.num_layers*(1+self.bidirectional))])
                decoder_hidden = encoder_hidden_reshaped
            else:
                decoder_hidden = encoder_hidden

        for i in range(MAX_LENGTH):
            if self.type == 'LSTM':
                decoder_output, (decoder_hidden, decoder_cell),attention_weights = self.call(decoder_input, (decoder_hidden, decoder_cell),encoder_outputs)
            else:
                decoder_output, decoder_hidden,attention_weights = self.call(decoder_input, decoder_hidden,encoder_outputs)
            decoder_outputs.append(decoder_output)
            attentions.append(attention_weights)

            if target_tensor is not None and teacher_ratio > random.random():
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)                
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions,dim=1)
        return decoder_outputs, decoder_hidden,attentions

    def call(self,input,hidden,outputs):
        
        embed = self.embedding(input)
        active_embed = torch.nn.functional.relu(embed)
        if isinstance(self.cell, torch.nn.LSTM):
            hidden_state, cell_state = hidden
            query = hidden_state.permute(1, 0, 2)
            context, attn_weights = self.attention_module(query, outputs)
            active_embed = torch.cat((active_embed, context), dim=2)
            output, (hidden_state, cell_state) = self.cell(
                active_embed, (hidden_state, cell_state))
            output = self.dropout(output)
            output = self.out(output)
            return output, (hidden_state, cell_state), attn_weights
        else:
            query = hidden.permute(1, 0, 2)
            context, attn_weights = self.attention_module(query, outputs)
            active_embed = torch.cat((active_embed, context), dim=2)
            output, hidden_state = self.cell(active_embed, hidden)
            output = self.dropout(output)
            output = self.out(output)
            return output, hidden_state, attn_weights

class Attention_Network(torch.nn.Module):
    def __init__(self,encoder:Encoder, decoder:Decoder_Attention):
        super(Attention_Network,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,input_tensor,target_tensor=None,teacher_ratio=0.2):
        input_tensor = input_tensor.to(DEVICE)
        if target_tensor is not None:
            target_tensor = target_tensor.to(DEVICE)
        encoder_outputs,encoder_hidden = self.encoder(input_tensor)
        decoder_outputs,_,attention_weights = self.decoder(encoder_outputs,encoder_hidden,target_tensor,teacher_ratio)
        return decoder_outputs,attention_weights
    
    def evaluate_data(self,dataloader,input_lang,output_lang,name='',heatmap=False):
    
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            datas = []
            for ind, data in enumerate(dataloader):
                input, output = data
                input = input.to(DEVICE)
                encoder_outputs, encoder_hidden = self.encoder(input)
                decoder_outputs, hidden_states,attention_weights = self.decoder(
                    encoder_outputs, encoder_hidden)

                _, topi = decoder_outputs.topk(1)
                decoded_ids = topi.squeeze()  # type: torch.Tensor

                for i in range(input.shape[0]):
                    input_word = []
                    predicted_word = []
                    output_word = []

                    for j in range(MAX_LENGTH):
                        if input[i][j].item() == end:
                            break
                        input_word.append(
                            input_lang.index_to_word[input[i][j].item()])
                    for j in range(MAX_LENGTH):
                        if decoded_ids[i][j].item() == end:
                            break
                        predicted_word.append(
                            output_lang.index_to_word[decoded_ids[i][j].item()])
                    for j in range(MAX_LENGTH):
                        if output[i][j].item() == end:
                            break
                        output_word.append(
                            output_lang.index_to_word[output[i][j].item()])
                    
                    if heatmap:
                        input_len = len(input_word)
                        output_len = len(output_word)
                        
                        
                        att_w = attention_weights[i,:output_len,:input_len].cpu().detach().numpy()
                        
                        sns.heatmap(att_w)
                        plt.xticks(range(output_len),output_word)
                        plt.yticks(range(input_len),input_word)
                        # plt.show()
                        plt.savefig("attention_heatmap_"+name+'_'+str(ind)+'.png')
                        plt.close()

                    input_word = ''.join(input_word)
                    predicted_word = ''.join(predicted_word)
                    output_word = ''.join(output_word)
                    datas.append([input_word, predicted_word, output_word])

        with open('predicted_attention/'+name+'.txt', 'w') as f:
            for item in datas:
                for datum in item:
                    f.write("%s,\t" % datum)
                f.write("\n")
        return data

    def test_loss_acc(self,encoder_model,decoder_model,criterion,dataloader,teacher_ratio):
   
        encoder_model.eval()
        decoder_model.eval()

        losses = []
        accuracies = []
        with torch.no_grad():
            for data in dataloader:
                input, output = data

                input = input.to(DEVICE)
                target = output.to(DEVICE)
                encoder_outputs, encoder_hidden = encoder_model(input)
                decoder_outputs, _ ,attention_weights = decoder_model(
                    encoder_outputs, encoder_hidden, target, teacher_ratio)
                loss = criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)),
                    target.view(-1)
                )
                # find the accuracy among decoder outputs and target
                word_accuracy = ((decoder_outputs.argmax(-1) == target).all(1).sum() /
                                decoder_outputs.size(0)).item()

                losses.append(loss.item())
                accuracies.append(word_accuracy)
        return np.mean(losses), np.mean(accuracies)

    def train_model(self,train_loader,valid_loader,input_lang,output_lang,test_loader=None,epochs=30,wandb_log=False,learning_rate=0.01,teacher_ratio=0.5,evaluate_test=False,heatmap=False):
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate, weight_decay=1e-5)           
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_optimizer, start_factor=1, end_factor=0.5, total_iters=epochs)
        decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_optimizer, start_factor=1, end_factor=0.5, total_iters=epochs)
        if wandb_log:
            wandb.login()
        for epoch in range(epochs):
            self.encoder.train()
            self.decoder.train()
            epoch_loss, epoch_acc = [], []

            for input, target in train_loader:
                input, target = input.to(DEVICE), target.to(DEVICE)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                outputs,attention_weights = self(input, target, teacher_ratio=teacher_ratio)

                loss = criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
                acc = ((outputs.argmax(-1) == target).all(1).sum() / target.size(0)).item()
                loss.backward()

                encoder_optimizer.step()
                decoder_optimizer.step()                
                epoch_loss.append(loss.item())
                epoch_acc.append(acc)

            train_loss, train_acc = np.mean(epoch_loss), np.mean(epoch_acc)

            valid_loss, valid_acc = self.validate_model(valid_loader, criterion)

            print(f'Epoch {epoch} | Train_Loss: {train_loss:.4f} | Train_Acc: {train_acc:.4f} | Valid_Loss: {valid_loss:.4f} | Valid_Acc: {valid_acc:.4f}')

            if wandb_log:
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': valid_loss,
                    'val_acc': valid_acc
                })

            encoder_scheduler.step()
            decoder_scheduler.step()
            if evaluate_test == True:
                if epoch % 10 == 0:
                    if test_loader is not None:
                        test_loss,test_acc = self.test_loss_acc(encoder_model=self.encoder,decoder_model=self.decoder,dataloader=test_loader,teacher_ratio=0.5,criterion=criterion) 
                        print("Test loss : {} | Test acc : {}".format(test_loss,test_acc))
        
                        self.evaluate_data(input_lang=input_lang,output_lang=output_lang,name='attention',dataloader=test_loader,heatmap=heatmap)
            
            torch.cuda.empty_cache()
        if evaluate_test == True:
            if test_loader is not None:
                test_loss,test_acc = self.test_loss_acc(encoder_model=self.encoder,decoder_model=self.decoder,dataloader=test_loader,teacher_ratio=0.5,criterion=criterion) 
                print("Test loss : {} | Test acc : {}".format(test_loss,test_acc))

                self.evaluate_data(input_lang=input_lang,output_lang=output_lang,name='attention',dataloader=test_loader,heatmap=heatmap) 
    
    def validate_model(self, dataloader, criterion):
        self.encoder.eval()
        self.decoder.eval()

        losses, accuracies = [], []
        with torch.no_grad():
            for input, target in dataloader:
                input, target = input.to(DEVICE), target.to(DEVICE)

                outputs,attention_weights = self(input, target, teacher_ratio=0)

                loss = criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
                acc = ((outputs.argmax(-1) == target).all(1).sum() / target.size(0)).item()

                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)