import tensorflow as tf
from config import *


class RNN_Seq2Seq:
    def __init__(self, num_encoder_tokens, num_decoder_tokens,
                 embedding_dim=64, latent_dim=128,
                 num_encoder_layers=1, num_decoder_layers=1,
                 cell_type='RNN'):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.cell_type = cell_type
        self.model = self.build_model()

    def build_model(self):

        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(None, self.num_encoder_tokens))
        x = tf.keras.layers.Dense(self.embedding_dim, activation='relu')(encoder_inputs)

        encoder_states = []
        for i in range(self.num_encoder_layers):
            return_seq = i < self.num_encoder_layers - 1
            rnn_cell = self.get_rnn_cell(return_seq)
            if self.cell_type == 'LSTM':
                x, state_h, state_c = rnn_cell(x)
                encoder_states = [state_h, state_c]
            else:
                x, state_h = rnn_cell(x)
                encoder_states = [state_h]

        # Decoder
        decoder_inputs = tf.keras.layers.Input(shape=(None, self.num_decoder_tokens))
        x_dec = tf.keras.layers.Dense(self.embedding_dim, activation='relu')(decoder_inputs)

        for i in range(self.num_decoder_layers):
            rnn_cell = self.get_rnn_cell(return_seq=True)
            if self.cell_type == 'LSTM':
                x_dec, _, _ = rnn_cell(x_dec, initial_state=encoder_states)
            else:
                x_dec, _ = rnn_cell(x_dec, initial_state=encoder_states)

        decoder_outputs = tf.keras.layers.Dense(self.num_decoder_tokens, activation='softmax')(x_dec)

        model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_rnn_cell(self, return_seq):
        """Returns the appropriate RNN cell type."""
        if self.cell_type == 'LSTM':
            return tf.keras.layers.LSTM(self.latent_dim, return_state=True, return_sequences=return_seq)
        elif self.cell_type == 'GRU':
            return tf.keras.layers.GRU(self.latent_dim, return_state=True, return_sequences=return_seq)
        else:
            return tf.keras.layers.SimpleRNN(self.latent_dim, return_state=True, return_sequences=return_seq)

    def train(self, train_encoder_input_data, train_decoder_input_data, train_decoder_target_data,
              val_encoder_input_data, val_decoder_input_data, val_decoder_target_data,
              batch_size=BATCH_SIZE, epochs=EPOCHS, save_model=False):
        self.model.fit(
            [train_encoder_input_data, train_decoder_input_data],
            train_decoder_target_data,
            validation_data=([val_encoder_input_data, val_decoder_input_data], val_decoder_target_data),
            batch_size=batch_size,
            epochs=epochs
        )
        if save_model:
            self.model.save('./models/rnn_model.h5')

