# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class RNN_Seq2Seq(nn.Module):
    def __init__(
        self,
        num_encoder_tokens: int,
        num_decoder_tokens: int,
        embedding_dim: int = 256,
        latent_dim: int = LATENT_DIM,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        cell_type: str = 'RNN',
    ):
        super().__init__()
        self.cell_type = cell_type.upper()
        # embedding layers, not one-hot → Linear
        self.encoder_embedding = nn.Embedding(num_encoder_tokens, embedding_dim)
        self.decoder_embedding = nn.Embedding(num_decoder_tokens, embedding_dim)

        rnn_cls = {
            'LSTM': nn.LSTM,
            'GRU':  nn.GRU,
            'RNN':  nn.RNN
        }[self.cell_type]

        self.encoder_rnn = rnn_cls(
            embedding_dim, latent_dim,
            num_layers=num_encoder_layers,
            batch_first=True
        )
        self.decoder_rnn = rnn_cls(
            embedding_dim, latent_dim,
            num_layers=num_decoder_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(latent_dim, num_decoder_tokens)

    def forward(self, enc_seq, dec_seq):
        """
        enc_seq: (B, T_enc) of encoder token‐indices
        dec_seq: (B, T_dec) of decoder token‐indices
        """
        # embed
        x_enc = self.encoder_embedding(enc_seq)  # (B, T_enc, E)
        x_dec = self.decoder_embedding(dec_seq)  # (B, T_dec, E)

        # encode
        _, enc_states = self.encoder_rnn(x_enc)
        # decode
        dec_outputs, _ = self.decoder_rnn(x_dec, enc_states)
        logits = self.output_layer(dec_outputs)       # (B, T_dec, V)
        return F.log_softmax(logits, dim=-1)
