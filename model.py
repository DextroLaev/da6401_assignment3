"""
model.py

Defines the vanilla sequence-to-sequence architecture components:

- `cell`: Utility to select the RNN cell class.
- `Encoder`: Embeds inputs and processes them through an RNN/LSTM/GRU.
- `Decoder`: Generates outputs step by step, with optional teacher forcing.
- `BeamSearchDecoder`: Extends `Decoder` to support beam search or Gumbel decoding.
- `Seq2Seq_Model`: Wraps encoder + decoder, and provides training, evaluation,
  and testing routines.

This module handles all core modeling logic for the vanilla (non-attention) setup.
"""


import torch
from config import *
import random
from dataset import start,end
import numpy as np
import wandb


torch.set_grad_enabled(True)

def cell(cell_type='RNN'):
    """
    Return the appropriate PyTorch RNN class based on the given cell type string.

    Args:
        cell_type (str): One of 'RNN', 'LSTM', or 'GRU'. Defaults to 'RNN'.

    Returns:
        torch.nn.Module: The corresponding recurrent layer class.
    """
    if cell_type == 'RNN':
        return torch.nn.RNN
    elif cell_type == 'LSTM':
        return torch.nn.LSTM
    elif cell_type == 'GRU':
        return torch.nn.GRU
    
class Encoder(torch.nn.Module):
    """
    Encoder module for a sequence-to-sequence model.

    Embeds the input token indices and processes them through a multi-layer
    RNN/LSTM/GRU to produce encoder outputs and the final hidden state.

    Args:
        cell_type (str): Type of recurrent cell ('RNN', 'LSTM', or 'GRU').
        num_layers (int): Number of stacked recurrent layers.
        hidden_dim (int): Dimensionality of the hidden state.
        embed_dim (int): Dimensionality of the input embeddings.
        input_dim (int): Size of the input vocabulary.
        dropout_rate (float): Dropout probability between layers (ignored if num_layers<=1).
        bidirectional (bool): If True, use a bidirectional RNN.
        batch_first (bool): If True, input/output tensors are batch-first.
    """

    def __init__(self,cell_type=TYPE,num_layers=ENCODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,embed_dim=EMBED_DIM,input_dim=INPUT_DIM,dropout_rate=DROPOUT_RATE,
                 bidirectional=BIDIRECTIONAL,batch_first=BATCH_FIRST):
        super(Encoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers        
        self.batch_first = batch_first
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.dropout_rate = 0 if num_layers <= 1 else dropout_rate

        self.embedding = torch.nn.Embedding(self.input_dim,self.embed_dim)
        self.cell = cell(self.cell_type)(self.embed_dim,self.hidden_dim,num_layers=self.num_layers,batch_first=batch_first,dropout=self.dropout_rate,bidirectional=bidirectional)
    
    def forward(self,input_tensor):
        """
        Run the encoder on input sequences.

        Args:
            input_tensor (Tensor): LongTensor of shape (batch_size, seq_len)
                containing input token indices.

        Returns:
            Tuple[Tensor, Tensor or Tuple]: 
              - outputs: Tensor of shape (batch_size, seq_len, hidden_dim * num_directions)
                containing all hidden states.
              - hidden: final hidden state (and cell state for LSTM) ready for the decoder.
        """

        encoder_hidden = torch.zeros(self.num_layers*(1+self.bidirectional),input_tensor.size(0),self.hidden_dim,device=DEVICE)
        encoder_cell = torch.zeros(self.num_layers*(1+self.bidirectional),input_tensor.size(0),self.hidden_dim,device=DEVICE)        

        if self.cell_type == 'LSTM':
            encoder_outputs,(encoder_hidden,encoder_cell) = self.forward_step(input_tensor,(encoder_hidden,encoder_cell))        
        else:
            encoder_outputs,encoder_hidden = self.forward_step(input_tensor,encoder_hidden)            

        if self.cell_type == 'LSTM':
            encoder_hidden = (encoder_hidden,encoder_cell)
        return encoder_outputs,encoder_hidden            

    def forward_step(self,input,hidden):
        '''
        This is a helper function that is used in the actual forward() function.
        '''
        embedded = self.embedding(input)
        embedded = torch.nn.functional.relu(embedded)
        
        if self.cell_type == 'LSTM':
            hidden_state, cell_state = hidden
            output, (hidden_state, cell_state) = self.cell(
                embedded, (hidden_state, cell_state))
            hidden_state = (hidden_state, cell_state)
        else:
            output, hidden_state = self.cell(embedded, hidden)
        

        return output, hidden_state

class Decoder(torch.nn.Module):
    """
    Decoder module for a sequence-to-sequence model without attention.

    Generates output tokens step-by-step, optionally using teacher forcing.

    Args:
        type (str): Type of recurrent cell ('RNN', 'LSTM', or 'GRU').
        num_layers (int): Number of stacked recurrent layers.
        hidden_dim (int): Dimensionality of the hidden state.
        dropout_rate (float): Dropout probability between layers.
        bidirectional (bool): If True, uses bidirectional recurrence internally.
        batch_first (bool): If True, input/output tensors are batch-first.
        embed_dim (int): Dimensionality of the token embeddings.
        output_dim (int): Size of the output vocabulary.
    """

    def __init__(self,type= TYPE,num_layers=DECODER_NUM_LAYERS,hidden_dim=HIDDEN_DIM,dropout_rate=DROPOUT_RATE,bidirectional=BIDIRECTIONAL,
                 batch_first=BATCH_FIRST,embed_dim=EMBED_DIM,output_dim=OUTPUT_DIM):
        super(Decoder, self).__init__()
        self.type = type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.dropout_rate = 0 if num_layers <= 1 else dropout_rate
        self.bidirectional = bidirectional

        self.embedding = torch.nn.Embedding(self.output_dim, self.embed_dim)
        self.cell = cell(type)(self.embed_dim, self.hidden_dim, num_layers=num_layers, batch_first=batch_first, dropout=self.dropout_rate, bidirectional=bidirectional)
        self.out = torch.nn.Linear(self.hidden_dim*(1+self.bidirectional), self.output_dim)

    def forward(self,encoder_outputs,encoder_hidden,target_tensor=None,teacher_ratio=0.5):
        """
        Decode a full sequence given encoder outputs and initial hidden state.

        Args:
            encoder_outputs (Tensor): Outputs from the encoder (unused here).
            encoder_hidden: Final hidden (and cell) state from the encoder.
            target_tensor (Tensor, optional): Ground-truth token indices for teacher forcing.
            teacher_ratio (float): Probability of using teacher forcing at each step.

        Returns:
            Tuple[Tensor, Tensor]:
              - decoder_outputs: Tensor of shape (batch_size, max_len, output_dim)
                with token logits.
              - hidden: final hidden (and cell) state.
        """

        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size,1,dtype=torch.long,device=DEVICE).fill_(start)
        decoder_outputs = []
        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            if encoder_hidden.shape[0] != self.num_layers*(1+self.bidirectional):
                encoder_hidden = encoder_hidden.mean(0)
                encoder_hidden = torch.stack([encoder_hidden for i in range(self.num_layers*(1+self.bidirectional))])
                encoder_cell = encoder_cell.mean(0)
                encoder_cell = torch.stack([encoder_cell for i in range(self.num_layers*(1+self.bidirectional))])
            decoder_cell = encoder_cell
            decoder_hidden = encoder_hidden
        else:
            if encoder_hidden.shape[0] != self.num_layers*(1+self.bidirectional):
                encoder_hidden = encoder_hidden.mean(0)
                encoder_hidden = torch.stack([encoder_hidden for i in range(self.num_layers*(1+self.bidirectional))])
            decoder_hidden = encoder_hidden

        for i in range(MAX_LENGTH):
            if self.type == 'LSTM':
                decoder_output, (decoder_hidden, decoder_cell) = self.forward_step(decoder_input, (decoder_hidden, decoder_cell))
            else:
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None and teacher_ratio > random.random():
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                _, topi = decoder_output.topk(1)                
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden

    def forward_step(self,input,hidden):
        
        embed = self.embedding(input)
        active_embed = torch.nn.functional.relu(embed)
        if isinstance(self.cell, torch.nn.LSTM):
            hidden_state, cell_state = hidden
            output, (hidden_state, cell_state) = self.cell(
                active_embed, (hidden_state, cell_state))            
            output = self.out(output)
            return output, (hidden_state, cell_state)
        else:
            output, hidden_state = self.cell(active_embed, hidden)
            output = self.out(output)
            return output, hidden_state
        
import heapq

def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    gumbel_noise = -torch.empty_like(logits).exponential_().log()
    y = torch.nn.functional.softmax((logits + gumbel_noise) / temperature, dim=-1)

    if hard:
        # Straight-through: forward as one-hot, backward as soft
        index = y.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        y = (y_hard - y).detach() + y
    return y

class BeamSearchDecoder(Decoder):
    """
    Decoder with beam search decoding (or Gumbel sampling if beam_size==1).

    Inherits from `Decoder` but overrides forward to perform batched beam search
    during inference when no teacher-forcing targets are provided.

    Args:
        beam_size (int): Number of beams to keep during search.
        use_gumbel (bool): If True and beam_size==1, use Gumbel-softmax sampling.
        temperature (float): Temperature for Gumbel-softmax.
        *args, **kwargs: Passed to base `Decoder` init.
    """
    def __init__(self, beam_size=3, use_gumbel=False, temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beam_size = beam_size
        self.use_gumbel = use_gumbel
        self.temperature = temperature
        

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_ratio=0.5):
        """
        Run beam search or fallback to greedy/Gumbel decoding.

        Args:
            encoder_outputs (Tensor): Encoder outputs (unused here).
            encoder_hidden: Final hidden (and cell) state from the encoder.
            target_tensor (Tensor, optional): Ground-truth for teacher forcing.
            teacher_ratio (float): Teacher forcing ratio when using greedy/Gumbel.

        Returns:
            Tuple[Tensor, Tensor]: 
              - decoded one-hot or logits of shape (batch_size, seq_len, output_dim)
              - final hidden state.
        """
        if self.beam_size == 1 or target_tensor is not None:
            return self.greedy_or_gumbel_decode(encoder_outputs, encoder_hidden, target_tensor, teacher_ratio)

        # Batched beam search for inference
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size * self.beam_size, 1), start, dtype=torch.long, device=DEVICE)

        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            decoder_hidden = self._expand_for_beam(self._match_hidden_shape(encoder_hidden), self.beam_size)
            decoder_cell = self._expand_for_beam(self._match_hidden_shape(encoder_cell), self.beam_size)
        else:
            decoder_hidden = self._expand_for_beam(self._match_hidden_shape(encoder_hidden), self.beam_size)
            decoder_cell = None

        sequences = torch.full((batch_size * self.beam_size, 1), start, dtype=torch.long, device=DEVICE)
        scores = torch.zeros(batch_size * self.beam_size, device=DEVICE)
        is_finished = torch.zeros_like(scores, dtype=torch.bool)

        for _ in range(MAX_LENGTH):
            if self.type == 'LSTM':
                output, (decoder_hidden, decoder_cell) = self.forward_step(decoder_input, (decoder_hidden, decoder_cell))
            else:
                output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)

            log_probs = torch.nn.functional.log_softmax(output[:, -1], dim=-1)
            vocab_size = log_probs.size(1)
            next_scores, next_tokens = torch.topk(log_probs, self.beam_size, dim=-1)

            # Expand to compute all beam combinations
            scores = scores.view(batch_size, self.beam_size, 1)
            next_scores = next_scores.view(batch_size, self.beam_size, self.beam_size)
            total_scores = scores + next_scores

            flat_scores = total_scores.view(batch_size, -1)
            top_scores, top_indices = torch.topk(flat_scores, self.beam_size, dim=-1)

            beam_indices = top_indices // self.beam_size
            token_indices = top_indices % self.beam_size

            new_sequences = []
            new_decoder_input = []
            new_hidden = []
            new_cell = [] if decoder_cell is not None else None

            for b in range(batch_size):
                for i in range(self.beam_size):
                    old_idx = b * self.beam_size + beam_indices[b, i]
                    token = next_tokens[old_idx][token_indices[b, i]]
                    seq = torch.cat([sequences[old_idx], token.unsqueeze(0)], dim=0)
                    new_sequences.append(seq)
                    new_decoder_input.append(token.view(1))
                    new_hidden.append(decoder_hidden[:, old_idx:old_idx+1])
                    if decoder_cell is not None:
                        new_cell.append(decoder_cell[:, old_idx:old_idx+1])

            sequences = torch.stack(new_sequences)
            decoder_input = torch.stack(new_decoder_input).to(DEVICE)
            decoder_hidden = torch.cat(new_hidden, dim=1)
            if decoder_cell is not None:
                decoder_cell = torch.cat(new_cell, dim=1)

            scores = top_scores.view(-1)
            is_finished = is_finished | (sequences[:, -1] == end)
            if is_finished.all():
                break

        sequences = sequences.view(batch_size, self.beam_size, -1)
        final_scores = scores.view(batch_size, self.beam_size)
        best_indices = final_scores.argmax(dim=1)
        best_sequences = sequences[torch.arange(batch_size), best_indices]

        one_hot_outputs = torch.nn.functional.one_hot(best_sequences[:, 1:], num_classes=self.output_dim).float()
        return one_hot_outputs, decoder_hidden
    
    def greedy_or_gumbel_decode(self, encoder_outputs, encoder_hidden, target_tensor=None, teacher_ratio=0.5):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.full((batch_size, 1), start, dtype=torch.long, device=DEVICE)
        decoder_outputs = []

        if self.type == 'LSTM':
            encoder_hidden, encoder_cell = encoder_hidden
            decoder_hidden = self._match_hidden_shape(encoder_hidden)
            decoder_cell = self._match_hidden_shape(encoder_cell)
        else:
            decoder_hidden = self._match_hidden_shape(encoder_hidden)
            decoder_cell = None

        for i in range(MAX_LENGTH):
            if self.type == 'LSTM':
                decoder_output, (decoder_hidden, decoder_cell) = self.forward_step(decoder_input, (decoder_hidden, decoder_cell))
            else:
                decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            use_teacher = target_tensor is not None and teacher_ratio > random.random()
            if use_teacher:
                decoder_input = target_tensor[:, i].unsqueeze(1)
            else:
                logits = decoder_output[:, -1]
                if self.use_gumbel:
                    gumbel_out = gumbel_softmax_sample(logits, temperature=self.temperature, hard=True)
                    token_ids = gumbel_out.argmax(dim=-1, keepdim=True)
                else:
                    token_ids = logits.argmax(dim=-1, keepdim=True)
                decoder_input = token_ids

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, decoder_hidden

    def _match_hidden_shape(self, hidden):
        if hidden.shape[0] != self.num_layers * (1 + self.bidirectional):
            mean = hidden.mean(0)
            return torch.stack([mean for _ in range(self.num_layers * (1 + self.bidirectional))])
        return hidden

    def _expand_for_beam(self, tensor, beam_size):
        if tensor.dim() == 3:  # [L, B, H]
            L, B, H = tensor.shape
            tensor = tensor.unsqueeze(2).repeat(1, 1, beam_size, 1)
            return tensor.view(L, B * beam_size, H)
        elif tensor.dim() == 2:  # [B, H]
            B, H = tensor.shape
            tensor = tensor.unsqueeze(1).repeat(1, beam_size, 1)
            return tensor.view(B * beam_size, H)
        elif tensor.dim() == 1:  # [B]
            B = tensor.shape[0]
            tensor = tensor.unsqueeze(1).repeat(1, beam_size)
            return tensor.view(B * beam_size)
        else:
            raise ValueError("Unsupported tensor shape for beam expansion")

class Seq2Seq_Model(torch.nn.Module):
    """
    Wraps an Encoder and Decoder into a full Seq2Seq model with training,
    validation, and testing routines.

    Args:
        encoder (Encoder): Instance of the Encoder class.
        decoder (Decoder): Instance of Decoder or BeamSearchDecoder.
    """
    def __init__(self,encoder:Encoder, decoder:Decoder):
        super(Seq2Seq_Model,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self,input_tensor,target_tensor=None,teacher_ratio=0.2):
        """
        Run a forward pass through encoder and decoder.

        Args:
            input_tensor (Tensor): Source token indices.
            target_tensor (Tensor, optional): Ground-truth token indices for teacher forcing.
            teacher_ratio (float): Probability of using teacher forcing.

        Returns:
            Tensor: Decoder logits of shape (batch_size, seq_len, vocab_size).
        """
        input_tensor = input_tensor.to(DEVICE)
        if target_tensor is not None:
            target_tensor = target_tensor.to(DEVICE)
        encoder_outputs,encoder_hidden = self.encoder(input_tensor)
        decoder_outputs,_ = self.decoder(encoder_outputs,encoder_hidden,target_tensor,teacher_ratio)
        return decoder_outputs
    
    def evaluate_dataset(self,encoder,decoder,dataloader,input_lang,output_lang,name=''):
        """
        This function is used to create a .txt file in which we write the input,predicted,ground_truth
        """
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            datas = []
            for ind, data in enumerate(dataloader):
                input, output = data
                input = input.to(DEVICE)
                encoder_outputs, encoder_hidden = encoder(input)
                decoder_outputs, _ = decoder(
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

                    input_word = ''.join(input_word)
                    predicted_word = ''.join(predicted_word)
                    output_word = ''.join(output_word)
                    datas.append([input_word, predicted_word, output_word])

        with open('predicted_vanilla/'+name+'.txt', 'w') as f:
            for item in datas:
                for datum in item:
                    f.write("%s,\t" % datum)
                f.write("\n")
        return data

    def test_loss_acc(self,encoder_model,decoder_model,dataloader,teacher_ratio):
        """
        This is a helper function used to calculate the loss and accuracy of the data passed
        Input:
            1. Encoder model
            2. Decoder model
            3. dataloader dataset
            4. teacher ratio
        Output:
            loss, accuracy
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
                decoder_outputs, _ = decoder_model(
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
        return np.mean(losses), np.mean(accuracies)

    def train_model(self,train_loader,valid_loader,input_lang,output_lang,test_loader=None,epochs=30,wandb_log=False,learning_rate=0.01,teacher_ratio=0.5,evaluate_test=False,save_model=False):
        """
        This function is used to train the Encoder-decoder Model
        Input:
            1. train_data_loader
            2. validation_data_loader
            3. input_lang
            4. output_lang
            5. test_data_loader
            6. epochs
            7. wandb_logging(TRUE/FALSE)
            8. Learning_rate
            9. teacher_ratio (for teacher forcing)
            10. evaluate_test (True/False) -> Making this true, make the model to evaluate the test data after every few epochs
            11. save_model (True/False) -> If you want to save the model, make it true
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
                outputs = self(input, target, teacher_ratio=teacher_ratio)

                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
                loss.backward()

                encoder_optimizer.step()
                decoder_optimizer.step()

                acc = ((outputs.argmax(-1) == target).all(1).sum() / target.size(0)).item()
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
        if evaluate_test == True:
            if test_loader is not None:
                test_loss,test_acc = self.test_loss_acc(encoder_model=self.encoder,decoder_model=self.decoder,dataloader=test_loader,teacher_ratio=0.5) 
                print("Test loss : {} | Test acc : {}".format(test_loss,test_acc))

                self.evaluate_dataset(encoder=self.encoder,decoder=self.decoder,input_lang=input_lang,output_lang=output_lang,name='vanilla',dataloader=test_loader) 

        if save_model:
            torch.save(self.state_dict(),'models/vanilla.pth')
            print('Model saved at directory models/')

    def validate_model(self, dataloader):
        """used for computing validation loss and accuracy.
            Input: Dataloader
            Output: loss, accuracy 
        """
        self.encoder.eval()
        self.decoder.eval()

        losses, accuracies = [], []
        with torch.no_grad():
            for input, target in dataloader:
                input, target = input.to(DEVICE), target.to(DEVICE)

                outputs = self(input, target, teacher_ratio=0)

                loss = self.criterion(outputs.view(-1, outputs.size(-1)), target.view(-1))
                acc = ((outputs.argmax(-1) == target).all(1).sum() / target.size(0)).item()

                losses.append(loss.item())
                accuracies.append(acc)

        return np.mean(losses), np.mean(accuracies)


