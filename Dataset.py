# import numpy as np

# class Dataset:
#     def __init__(self,train_path,test_path,val_path):
#         self.train_path = train_path
#         self.test_path = test_path
#         self.val_path = val_path
    
#     def load_dataset(self,path):
#         input_texts = []
#         target_texts = []
#         input_characters = set()
#         target_characters = set()

#         with open(path,'r',encoding='utf-8') as fd:
#             c = fd.read().split('\n')
        
#         for line in c:
#             if len(line) != 0:
#                 input_text, target_text = line.split('\t')[:2]
#                 target_text = '\t' + target_text + '\n'

#                 input_texts.append(input_text)
#                 target_texts.append(target_text)
#                 for char in input_text:
#                     if char not in input_characters:
#                         input_characters.update(char)
#                 for char in target_text:
#                     if char not in target_characters:
#                         target_characters.update(char)
        
#         input_characters.add(" ")
#         target_characters.add(" ")

#         input_characters = sorted(list(input_characters))
#         target_characters = sorted(list(target_characters))

#         num_encoder_tokens = len(input_characters)
#         num_decoder_tokens = len(target_characters)

#         max_encoder_input_seq_length = max([len(txt) for txt in input_texts])
#         max_decoder_output_seq_length = max([len(txt) for txt in target_texts])

#         encoder_input_data = np.zeros((len(input_texts),max_encoder_input_seq_length,num_encoder_tokens),dtype='float32')
#         decoder_input_data = np.zeros((len(target_texts),max_decoder_output_seq_length,num_decoder_tokens),dtype='float32')
#         decoder_target_data = np.zeros((len(target_texts), max_decoder_output_seq_length, num_decoder_tokens),dtype="float32")

#         input_token_index = {char: i for i, char in enumerate(input_characters)}
#         target_token_index = {char: i for i, char in enumerate(target_characters)}

#         for i,(input_text,target_text) in enumerate(zip(input_texts,target_texts)):

#             for t, char in enumerate(input_text):
#                 encoder_input_data[i,t,input_token_index[char]] = 1.0
#             encoder_input_data[i,len(input_text):,input_token_index[" "]] = 1.0

#             for t, char in enumerate(target_text):
#                 decoder_input_data[i, t, target_token_index[char]] = 1.0
#                 if t > 0:
#                     decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
#             decoder_input_data[i, len(target_text):, target_token_index[" "]] = 1.0  
#             decoder_target_data[i, len(target_text)-1:, target_token_index[" "]] = 1.0

#         return {
#             "encoder_input_data": encoder_input_data,
#             "decoder_input_data": decoder_input_data,
#             "decoder_target_data": decoder_target_data,
#             "input_token_index": input_token_index,
#             "target_token_index": target_token_index,
#             "num_encoder_tokens": num_encoder_tokens,
#             "num_decoder_tokens": num_decoder_tokens,
#             "max_encoder_seq_length": max_encoder_input_seq_length,
#             "max_decoder_seq_length": max_decoder_output_seq_length
#         }
    

#     def load_train_data(self): return self.load_dataset(self.train_path)
#     def load_test_data(self): return self.load_dataset(self.test_path)
#     def load_val_data(self): return self.load_dataset(self.val_path)

# if __name__ == '__main__':
#     train_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
#     test_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv'
#     val_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'
#     dataset_c = Dataset(train_path,test_path,val_path)
#     dataset_c.load_train_data()
import numpy as np

class Dataset:
    def __init__(self, train_path, test_path, val_path):
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path

        # Shared character sets across all splits
        self.input_characters = set()
        self.target_characters = set()

        for path in [train_path, val_path, test_path]:
            with open(path, 'r', encoding='utf-8') as fd:
                lines = fd.read().split('\n')
                for line in lines:
                    if line.strip():
                        input_text, target_text = line.split('\t')[:2]
                        target_text = '\t' + target_text + '\n'
                        self.input_characters.update(input_text)
                        self.target_characters.update(target_text)

        # Add padding token and sort
        self.input_characters.add(" ")
        self.target_characters.add(" ")
        self.input_characters = sorted(list(self.input_characters))
        self.target_characters = sorted(list(self.target_characters))

        # Token index maps
        self.input_token_index = {char: i for i, char in enumerate(self.input_characters)}
        self.target_token_index = {char: i for i, char in enumerate(self.target_characters)}

        # Token counts
        self.num_encoder_tokens = len(self.input_characters)
        self.num_decoder_tokens = len(self.target_characters)

    def load_dataset(self, path):
        input_texts = []
        target_texts = []

        with open(path, 'r', encoding='utf-8') as fd:
            lines = fd.read().split('\n')
            for line in lines:
                if line.strip():
                    input_text, target_text = line.split('\t')[:2]
                    target_text = '\t' + target_text + '\n'
                    input_texts.append(input_text)
                    target_texts.append(target_text)

        max_encoder_seq_length = max(len(txt) for txt in input_texts)
        max_decoder_seq_length = max(len(txt) for txt in target_texts)

        encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
        decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')
        decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, self.num_decoder_tokens), dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.0
            encoder_input_data[i, len(input_text):, self.input_token_index[" "]] = 1.0  # Padding

            for t, char in enumerate(target_text):
                decoder_input_data[i, t, self.target_token_index[char]] = 1.0
                if t > 0:
                    decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.0
            decoder_input_data[i, len(target_text):, self.target_token_index[" "]] = 1.0
            decoder_target_data[i, len(target_text) - 1:, self.target_token_index[" "]] = 1.0

        return {
            "encoder_input_data": encoder_input_data,
            "decoder_input_data": decoder_input_data,
            "decoder_target_data": decoder_target_data,
            "input_token_index": self.input_token_index,
            "target_token_index": self.target_token_index,
            "num_encoder_tokens": self.num_encoder_tokens,
            "num_decoder_tokens": self.num_decoder_tokens,
            "max_encoder_seq_length": max_encoder_seq_length,
            "max_decoder_seq_length": max_decoder_seq_length
        }

    def load_train_data(self): return self.load_dataset(self.train_path)
    def load_val_data(self): return self.load_dataset(self.val_path)
    def load_test_data(self): return self.load_dataset(self.test_path)
