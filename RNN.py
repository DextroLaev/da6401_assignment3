import tensorflow as tf
from config import *


class RNN_Seq2Seq:
    def __init__(self,num_encoder_tokens,num_decoder_tokens,
                 embedding_dim=64,
                 latent_dim=128,
                 num_layers=1
                 ):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.model = self.build_model()
    
    def build_model(self):

        # Encoder
        encoder_inputs = tf.keras.layers.Input(shape=(None,self.num_encoder_tokens))
        x = tf.keras.layers.Dense(self.embedding_dim,activation='relu')(encoder_inputs)

        for i in range(self.num_layers):
            return_seq = i < self.num_layers-1
            rnn = tf.keras.layers.SimpleRNN(self.latent_dim,return_state=True,return_sequences=return_seq)
            x,state_h = rnn(x)

        encoder_states = state_h

        # Decoder

        decoder_inputs = tf.keras.layers.Input(shape=(None,self.num_decoder_tokens))

        x_dec = tf.keras.layers.Dense(self.embedding_dim, activation='relu')(decoder_inputs)

        for i in range(self.num_layers):
            rnn = tf.keras.layers.SimpleRNN(self.latent_dim,return_state=True,return_sequences=True)
            x_dec,_ = rnn(x_dec,initial_state=encoder_states)            

        decoder_outputs = tf.keras.layers.Dense(self.num_decoder_tokens, activation='softmax')(x_dec)
        model = tf.keras.models.Model([encoder_inputs,decoder_inputs],decoder_outputs)
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        return model
    
    def train(self,train_encoder_input_data,train_decoder_input_data,train_decoder_target_data,
              val_encoder_input_data,val_decoder_input_data,val_decoder_target_data,
              batch_size=BATCH_SIZE,epochs=EPOCHS,save_model=False):
        self.model.fit(
            [train_encoder_input_data,train_decoder_input_data],
            train_decoder_target_data,
            validation_data=([val_encoder_input_data,val_decoder_input_data],val_decoder_target_data),            
            # validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs
        )
        if save_model:
            self.model.save('./models/rnn_model.h5')