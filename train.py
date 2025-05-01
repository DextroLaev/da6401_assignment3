from Dataset import Dataset
from config import *
from RNN import RNN_Seq2Seq

if __name__ == '__main__':
    train_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
    test_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv'
    val_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'
    dataset_c = Dataset(train_path,test_path,val_path)


    train_data_info = dataset_c.load_train_data()
    val_data_info = dataset_c.load_val_data()


    seq2seq_rnn = RNN_Seq2Seq(num_encoder_tokens=train_data_info['num_encoder_tokens'],
                              num_decoder_tokens=train_data_info['num_decoder_tokens'],
                              latent_dim=LATENT_DIM,embedding_dim=256,num_layers=3)
    
    seq2seq_rnn.train(train_encoder_input_data=train_data_info['encoder_input_data'],
                      train_decoder_input_data=train_data_info['decoder_input_data'],
                      train_decoder_target_data=train_data_info['decoder_target_data'],
                      val_encoder_input_data=val_data_info['encoder_input_data'],
                      val_decoder_input_data=val_data_info['decoder_input_data'],
                      val_decoder_target_data=val_data_info['decoder_target_data'],
                      batch_size=BATCH_SIZE,epochs=EPOCHS,save_model=True
                      )