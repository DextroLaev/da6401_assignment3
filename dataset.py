import torch
import numpy as np
import pandas as pd
import random
from config import *
import os

start = 0
end = 1

class Corpus:

    def __init__(self, language='hi'):
        self.language = language
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {0: 'SOS', 1: 'EOS'}
        self.n_letters = 2

    def add_word(self, word):
        for letter in str(word):
            self.add_letter(letter)

    def add_letter(self, letter):
        if letter in self.word_to_index:
            self.word_to_count[letter] += 1
        else:
            self.word_to_index[letter] = self.n_letters
            self.word_to_count[letter] = 1
            self.index_to_word[self.n_letters] = letter
            self.n_letters += 1

class Dataset:
    def __init__(self):
        pass

    def read_file(self,lang='hi',type='train'):
        path = './dakshina_dataset_v1.0/{}/lexicons/{}.translit.sampled.{}.tsv'.format(lang,lang,type)
        data = pd.read_csv(path,header=None,sep='\t')
        return np.array(data[1]),np.array(data[0])
    
    def set_words(self,lang='hi',type='train'):
        input_lang, output_lang = Corpus(lang),Corpus('eng')
        input_words, output_words = self.read_file(lang, type)
        word_pairs = [[input_words[i], output_words[i]]
                    for i in range(len(input_words))]
        for word in input_words:
            input_lang.add_word(word)
        for word in output_words:
            output_lang.add_word(word)
        return input_lang, output_lang, word_pairs
    
    def word_to_index(self,lang,word):
        return [lang.word_to_index[char] for char in str(word)] 

    def convert_tensor(self,lang,word):
        indexes = self.word_to_index(lang, word)
        indexes.append(end)
        return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)
    
    def get_pair(self,input_lang,output_lang,pair):
        input_tensor = self.convert_tensor(input_lang, pair[0])
        target_tensor = self.convert_tensor(output_lang, pair[1])
        return (input_tensor, target_tensor)
    
    def create_dataLoader(self,pairs,batch_size):
        
        n = len(pairs)
        input_ids_train = np.ones((n, MAX_LENGTH), dtype=np.int32)
        target_ids_train = np.ones((n, MAX_LENGTH), dtype=np.int32)

        for idx, (inp, tgt) in enumerate(pairs):
            inp_ids = self.word_to_index(self.input_lang, inp)
            tgt_ids = self.word_to_index(self.output_lang, tgt)
            inp_ids.append(end)
            tgt_ids.append(end)
            input_ids_train[idx, :len(inp_ids)] = inp_ids
            target_ids_train[idx, :len(tgt_ids)] = tgt_ids

        data = torch.utils.data.TensorDataset(torch.LongTensor(input_ids_train).to(DEVICE),
                                                    torch.LongTensor(target_ids_train).to(DEVICE))

        sampler = torch.utils.data.RandomSampler(data)
        dataloader = torch.utils.data.DataLoader(
            data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def load_data(self,batch_size,lang='hi'):
        self.input_lang,self.output_lang,train_pairs = self.set_words(lang=lang,type='train')
        train_dataloader = self.create_dataLoader(batch_size=batch_size,pairs=train_pairs)
        _,_,test_pairs = self.set_words(lang=lang,type='test')
        test_dataloader = self.create_dataLoader(batch_size=batch_size,pairs=test_pairs)
        _,_,val_pairs = self.set_words(lang=lang,type='dev')
        val_dataloader = self.create_dataLoader(batch_size=batch_size,pairs=val_pairs)
        return self.input_lang,self.output_lang,train_dataloader,test_dataloader,val_dataloader
    
if __name__ == '__main__':
    dataset_c = Dataset()
    input_lang, output_lang, pairs = dataset_c.set_words('hi','train')
    print(random.choice(pairs))
    print("Number of words in input language: ", len(pairs))
    print("Number of characters in input language: ", input_lang.n_letters)
    print("Number of characters in output language: ", output_lang.n_letters)
    training_pairs = [dataset_c.get_pair(
        input_lang, output_lang, pair) for pair in pairs]
    print("Training pairs size: ", len(training_pairs))
    print(input_lang.word_to_index)
    print((output_lang.word_to_index))
    test_input,test_output,train_loader,test_loader,val_loader = dataset_c.load_data(batch_size=32,lang='hi')
    print(test_input.word_to_index)
    print(test_output.word_to_index)
    print(test_loader)
