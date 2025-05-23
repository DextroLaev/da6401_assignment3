"""
AttentionModel.py

Defines the attention‐augmented sequence‐to‐sequence architecture for transliteration tasks.

Components:

- `Attention_Module`: Computes context vectors via additive attention over encoder outputs.
- `Decoder_Attention`: Decoder that at each step attends to encoder outputs and generates one token.
- `Attention_Network`: High‐level wrapper that ties the encoder and attention decoder together,
  and provides `forward`, training, validation, and evaluation routines.

Each class and method is documented to explain its purpose, arguments, and return values.
"""

from model import Encoder,cell
import torch
from config import *
from dataset import start,end
import random
import numpy as np
import wandb
import seaborn as sns
import matplotlib.pyplot as plt

class Attention_Module(torch.nn.Module):
    """
    Additive attention mechanism (Bahdanau style).

    Computes attention scores between a decoder query and encoder key sequences,
    producing a context vector as their weighted sum.

    Args:
        layers (int): Number of decoder layers (for shaping query).
        hidden_dim (int): Dimensionality of each RNN hidden state.
        bidirectional (bool): Whether the encoder is bidirectional.

    Attributes:
        W_attn (Linear): Projects the decoder query.
        U_attn (Linear): Projects all encoder keys.
        V_attn (Linear): Scores combined projections to a scalar.
    """
    def __init__(self,layers,hidden_dim, bidirectional= False):
        super(Attention_Module,self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.layers = layers
        
        self.W_attn = torch.nn.Linear(hidden_dim*(1+bidirectional),hidden_dim*(1+bidirectional),bias=False)
        self.U_attn = torch.nn.Linear(hidden_dim*(1+bidirectional),hidden_dim*(1+bidirectional),bias=False)
        self.V_attn = torch.nn.Linear(hidden_dim*(1+bidirectional),1,bias=False)        

    def forward(self, query,keys):
        """
        Compute the context vector and attention weights.

        Args:
            query (Tensor): Decoder hidden states of shape (batch, seq=1, dim).
            keys  (Tensor): Encoder outputs of shape (batch, src_len, dim).

        Returns:
            context (Tensor): (batch, 1, dim) weighted sum of keys.
            attention_weights (Tensor): (batch, 1, src_len) normalized scores.
        """
        if self.bidirectional:
            query_t = torch.cat([query[:,-2,:],query[:,-1,:]],dim=1).unsqueeze(1)
            
        else:
            query_t = query[:,-1,:].unsqueeze()            
        query_proj = self.W_attn(query_t)
        
        keys_proj = self.U_attn(keys)
        energy = torch.tanh(query_proj+keys_proj)

        attention_score = self.V_attn(energy).squeeze(2)
        attention_weights = torch.nn.functional.softmax(attention_score,dim=1).unsqueeze(1)
        context = torch.bmm(attention_weights,keys)
        return context,attention_weights  
        

class Decoder_Attention(torch.nn.Module):
    """
    Decoder with Bahdanau attention.

    At each decoding step:
      1. Embeds the previous token.
      2. Computes attention context over all encoder outputs.
      3. Concatenates embedding + context and feeds through RNN cell.
      4. Emits logits over the output vocabulary.

    Args:
        type (str): RNN cell type ('RNN', 'LSTM', 'GRU').
        num_layers (int): Number of RNN layers.
        hidden_dim (int): Hidden state size.
        dropout_rate (float): Dropout probability between RNN layers.
        bidirectional (bool): Whether encoder was bidirectional.
        batch_first (bool): If True, tensors are (batch, seq, dim).
        embed_dim (int): Embedding size for input tokens.
        output_dim (int): Vocabulary size for outputs.
    """
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
        """
        Decode a full sequence, optionally using teacher forcing.

        Args:
            encoder_outputs (Tensor): (batch, src_len, dim) from the encoder.
            encoder_hidden: Final hidden (and cell) states from the encoder.
            target_tensor (Tensor, optional): Ground-truth token indices for teacher forcing.
            teacher_ratio (float): Probability to use the ground-truth token at each step.

        Returns:
            decoder_outputs (Tensor): (batch, max_len, output_dim) logits.
            decoder_hidden: Final hidden (and cell) states.
            attentions (Tensor): (batch, max_len, src_len) attention weights per step.
        """
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
        """
        Perform one decoding step with attention.

        Args:
            input_step (Tensor): (batch, 1) previous token indices.
            hidden: Current hidden (and cell) state.
            encoder_outputs (Tensor): All encoder states for attention.

        Returns:
            output (Tensor): (batch, 1, output_dim) logits for the next token.
            new_hidden: Updated hidden (and cell) state.
            attn_weights (Tensor): (batch, 1, src_len) attention weights.
        """
        embed = self.embedding(input)
        active_embed = torch.nn.functional.relu(embed)
        if isinstance(self.cell, torch.nn.LSTM):
            hidden_state, cell_state = hidden
            query = hidden_state.permute(1, 0, 2)

            # h_fwd = hidden_state[-2]
            # h_bwd = hidden_state[-1]
            # h_top = torch.cat([h_fwd,h_bwd],dim=1)
            # query = h_top.unsqueeze(1)
            context, attn_weights = self.attention_module(query, outputs)
            active_embed = torch.cat((active_embed, context), dim=2)
            output, (hidden_state, cell_state) = self.cell(
                active_embed, (hidden_state, cell_state))
            output = self.dropout(output)
            output = self.out(output)
            return output, (hidden_state, cell_state), attn_weights
        else:
            query = hidden.permute(1, 0, 2)
            # h_fwd = hidden_state[-2]
            # h_bwd = hidden_state[-1]
            # h_top = torch.cat([h_fwd,h_bwd],dim=1)            
            # query = h_top.unsqueeze(1)
            context, attn_weights = self.attention_module(query, outputs)
            active_embed = torch.cat((active_embed, context), dim=2)
            output, hidden_state = self.cell(active_embed, hidden)
            output = self.dropout(output)
            output = self.out(output)
            return output, hidden_state, attn_weights

class Attention_Network(torch.nn.Module):
    """
    Full seq2seq model combining an encoder and the attention decoder.

    Provides `forward`, training, validation, and data‐dump routines.

    Args:
        encoder (Encoder): Source encoder model instance.
        decoder (Decoder_Attention): Attention decoder instance.
    """
    def __init__(self,encoder:Encoder, decoder:Decoder_Attention):
        super(Attention_Network,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self,input_tensor,target_tensor=None,teacher_ratio=0.2):
        """
        Compute logits and attention weights for a batch.

        Args:
            input_tensor (Tensor): (batch, src_len) source token indices.
            target_tensor (Tensor, optional): (batch, tgt_len) for teacher forcing.
            teacher_ratio (float): Teacher forcing probability.

        Returns:
            decoder_outputs (Tensor): (batch, max_len, vocab) logits.
            attention_weights (Tensor): (batch, max_len, src_len).
        """
        input_tensor = input_tensor.to(DEVICE)
        if target_tensor is not None:
            target_tensor = target_tensor.to(DEVICE)
        encoder_outputs,encoder_hidden = self.encoder(input_tensor)
        decoder_outputs,_,attention_weights = self.decoder(encoder_outputs,encoder_hidden,target_tensor,teacher_ratio)
        return decoder_outputs,attention_weights
    
    def evaluate_data(self,dataloader,input_lang,output_lang,name='',heatmap=False):
        """
        Generate predictions over a DataLoader, optionally saving attention heatmaps.

        Args:
            dataloader (DataLoader): Batches of test examples.
            input_lang, output_lang: Corpora for mapping indices to chars.
            name (str): Prefix for output files.
            heatmap (bool): If True, saves per-example attention plots.
        """
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
                        plt.savefig("predicted_attention/attention_heatmap_"+name+'_'+str(ind)+'.png')
                        plt.close()

                    input_word = ''.join(input_word)
                    predicted_word = ''.join(predicted_word)
                    output_word = ''.join(output_word)
                    datas.append([input_word, predicted_word, output_word])
        torch.cuda.empty_cache()
        with open('predicted_attention/'+name+'.txt', 'w') as f:
            for item in datas:
                for datum in item:
                    f.write("%s,\t" % datum)
                f.write("\n")
        return data

    def test_loss_acc(self,encoder_model,decoder_model,dataloader,teacher_ratio):
        """
        Compute average test loss and sequence‐level accuracy.

        Args:
            encoder_model, decoder_model: Models to evaluate.
            dataloader (DataLoader): Test data.
            teacher_ratio (float): For decoder calls.

        Returns:
            Tuple[float, float]: (mean_loss, mean_sequence_accuracy)
        """
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
                loss = self.criterion(
                    decoder_outputs.view(-1, decoder_outputs.size(-1)),
                    target.view(-1)
                )
                # find the accuracy among decoder outputs and target
                word_accuracy = ((decoder_outputs.argmax(-1) == target).all(1).sum() /
                                decoder_outputs.size(0)).item()

                losses.append(loss.item())
                accuracies.append(word_accuracy)
        torch.cuda.empty_cache()
        return np.mean(losses), np.mean(accuracies)

    def train_model(self,train_loader,valid_loader,input_lang,output_lang,test_loader=None,epochs=30,wandb_log=False,learning_rate=0.01,teacher_ratio=0.5,evaluate_test=False,heatmap=False,save_model=False):
        """
        Full training loop over epochs, with optional W&B logging and test evaluation.

        Args:
            train_loader, valid_loader (DataLoader): Training and validation data.
            input_lang, output_lang: Corpora for mapping chars.
            test_loader (DataLoader, optional): For intermediate test eval.
            epochs (int): Number of training epochs.
            wandb_log (bool): If True, log metrics to W&B.
            learning_rate (float): Optimizer LR.
            teacher_ratio (float): Teacher forcing probability.
            evaluate_test (bool): If True, evaluate on test every 10 epochs.
            heatmap (bool): If True, save attention heatmaps post‐training.
            save_model (bool): If True, serialize the final model weights.
        """
        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate, weight_decay=1e-5)           
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate, weight_decay=1e-5)
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

                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
                acc = ((outputs.argmax(-1) == target).all(1).sum() / target.size(0)).item()
                loss.backward()

                encoder_optimizer.step()
                decoder_optimizer.step()                
                epoch_loss.append(loss.item())
                epoch_acc.append(acc)

            train_loss, train_acc = np.mean(epoch_loss), np.mean(epoch_acc)

            valid_loss, valid_acc = self.validate_model(valid_loader)

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
                        test_loss,test_acc = self.test_loss_acc(encoder_model=self.encoder,decoder_model=self.decoder,dataloader=test_loader,teacher_ratio=0.5) 
                        print("Test loss : {} | Test acc : {}".format(test_loss,test_acc))                        
            torch.cuda.empty_cache()
        if evaluate_test == True:
            if test_loader is not None:
                test_loss,test_acc = self.test_loss_acc(encoder_model=self.encoder,decoder_model=self.decoder,dataloader=test_loader,teacher_ratio=0.5) 
                print("Test loss : {} | Test acc : {}".format(test_loss,test_acc))

                self.evaluate_data(input_lang=input_lang,output_lang=output_lang,name='attention',dataloader=test_loader,heatmap=heatmap) 
        
        if save_model:
            torch.save(self.state_dict(),'models/attention_new.pth')
            print('Model saved at models/')
    
    def validate_model(self, dataloader):
        """
        Compute validation loss and accuracy without teacher forcing.

        Args:
            dataloader (DataLoader): Validation data.

        Returns:
            Tuple[float, float]: (mean_loss, mean_sequence_accuracy)
        """
        self.encoder.eval()
        self.decoder.eval()

        losses, accuracies = [], []
        with torch.no_grad():
            for input, target in dataloader:
                input, target = input.to(DEVICE), target.to(DEVICE)

                outputs,attention_weights = self(input, target, teacher_ratio=0)

                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
                acc = ((outputs.argmax(-1) == target).all(1).sum() / target.size(0)).item()

                losses.append(loss.item())
                accuracies.append(acc)
        torch.cuda.empty_cache()
        return np.mean(losses), np.mean(accuracies)