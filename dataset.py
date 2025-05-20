"""
dataset.py

Load and preprocess the Dakshina transliteration dataset for seq2seq training.

This module provides:

- `Corpus`: Builds character-level vocabularies for any given language.
- `Dataset`: Reads TSV files for train/dev/test splits, constructs `Corpus`
  instances, converts words to index sequences (with SOS/EOS tokens),
  and returns PyTorch DataLoaders ready for training or evaluation.

Usage:
    from dataset import Dataset
    input_lang, output_lang, train_loader, test_loader, valid_loader = \
        Dataset().load_data(batch_size=32, lang='hi')
"""

import torch
import numpy as np
import pandas as pd
import random
from config import *
import os

start = 0
end = 1

class Corpus:
    """
    Character-level vocabulary builder.

    Attributes:
        language (str): ISO code of the language (e.g., 'hi').
        word_to_index (dict): Maps each character to a unique integer index.
        word_to_count (dict): Tracks frequency of each character.
        index_to_word (dict): Reverse mapping from index to character.
        n_letters (int): Total number of unique tokens (including SOS and EOS).
    """

    def __init__(self, language='hi'):
        self.language = language
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {0: 'SOS', 1: 'EOS'}
        self.n_letters = 2

    def add_word(self, word):
        """
        Add all characters of a word to the vocabulary.

        Args:
            word (str): Input string whose characters to register.
        """
        for letter in str(word):
            self.add_letter(letter)

    def add_letter(self, letter):
        """
        Register a single character in the vocabulary.

        Increments count if already present, otherwise assigns a new index.

        Args:
            letter (str): Single-character string to add.
        """
        if letter in self.word_to_index:
            self.word_to_count[letter] += 1
        else:
            self.word_to_index[letter] = self.n_letters
            self.word_to_count[letter] = 1
            self.index_to_word[self.n_letters] = letter
            self.n_letters += 1

class Dataset:
    """
    Loader and converter for the Dakshina transliteration dataset.

    Provides methods to:
      - Read train/dev/test TSV files.
      - Build input and output `Corpus` vocabularies.
      - Convert words to index sequences with EOS appended.
      - Create padded PyTorch DataLoaders.

    Methods:
        read_file(lang, type) -> (src_array, tgt_array)
        set_words(lang, type) -> (input_corpus, output_corpus, word_pairs)
        word_to_index(lang, word) -> List[int]
        convert_tensor(lang, word) -> Tensor
        get_pair(input_lang, output_lang, pair) -> (Tensor, Tensor)
        create_dataLoader(pairs, batch_size) -> DataLoader
        load_data(batch_size, lang) -> (input_corpus, output_corpus, train_loader, test_loader, val_loader)
    """
    def __init__(self):
        pass

    def read_file(self,lang='hi',type='train'):
        """
        Read a TSV file for the specified split.

        Args:
            lang (str): Language code matching folder name.
            type (str): One of 'train', 'dev', or 'test'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of source and target words.
        """
        path = './dakshina_dataset_v1.0/{}/lexicons/{}.translit.sampled.{}.tsv'.format(lang,lang,type)
        data = pd.read_csv(path,header=None,sep='\t')
        return np.array(data[1]),np.array(data[0])
    
    def set_words(self,lang='hi',type='train'):
        """
        Build vocabularies and word pairs for a given data split.

        Args:
            lang (str): Language code for input.
            type (str): Data split: 'train', 'dev', or 'test'.

        Returns:
            Tuple[Corpus, Corpus, List[List[str]]]:
              - input_lang: Corpus for input characters
              - output_lang: Corpus for output characters
              - word_pairs: List of [source_word, target_word] pairs
        """
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
        """
        Map a word (string) to a list of character indices.

        Args:
            lang (Corpus): Character vocabulary.
            word (str): Word to convert.

        Returns:
            List[int]: Sequence of character indices.
        """
        return [lang.word_to_index[char] for char in str(word)] 

    def convert_tensor(self,lang,word):
        """
        Convert a word into a LongTensor with EOS appended.

        Args:
            lang (Corpus): Character vocabulary.
            word (str): Word to convert.

        Returns:
            Tensor: Shape (seq_len, 1) on DEVICE.
        """
        indexes = self.word_to_index(lang, word)
        indexes.append(end)
        return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1, 1)
    
    def get_pair(self,input_lang,output_lang,pair):
        """
        Get the input/output tensors for one word pair.

        Args:
            input_lang (Corpus): Source character vocabulary.
            output_lang (Corpus): Target character vocabulary.
            pair (List[str]): [source_word, target_word].

        Returns:
            Tuple[Tensor, Tensor]: input_tensor, target_tensor.
        """
        input_tensor = self.convert_tensor(input_lang, pair[0])
        target_tensor = self.convert_tensor(output_lang, pair[1])
        return (input_tensor, target_tensor)
    
    def create_dataLoader(self,pairs,batch_size):
        """
        Build a padded DataLoader from word pairs.

        Pads or truncates to MAX_LENGTH, adds EOS, and shuffles.

        Args:
            pairs (List[List[str]]): List of [src, tgt] word pairs.
            batch_size (int): Batch size.

        Returns:
            DataLoader: Yields (input_batch, target_batch).
        """
        
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
        """
        Prepare train/dev/test DataLoaders and their vocabularies.

        Args:
            batch_size (int): Batch size for all splits.
            lang (str): Language code.

        Returns:
            Tuple[Corpus, Corpus, DataLoader, DataLoader, DataLoader]:
              input_lang, output_lang, train_loader, test_loader, valid_loader
        """
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
