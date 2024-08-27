#%%
import string
import re

import torch
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
from collections import deque


class IEHyperionDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        self.df = pd.DataFrame()
        self.df['Testo'] = df['Testo'].map(clean_text)
        self.df['Stralci'] = df['Stralci'].map(lambda x: [clean_text(s) for s in x])
        self.df['Repertori'] = df['Repertori']
        self.df['Char_bounds'] = self.df.apply(lambda x: find_char_bounds(x['Stralci'], x['Testo']), axis=1).values.tolist()
        self.df['Bounds'] = self.df.apply(lambda x: find_word_bounds(x['Stralci'], x['Testo']), axis=1).values.tolist()
        self.df['Char_segmentation'] = self.df['Char_bounds'].map(find_segmentation_by_bounds)
        self.df['Segmentation'] = self.df['Bounds'].map(find_segmentation_by_bounds)
        
        self.encodings = self.tokenizer(self.df['Testo'].tolist(),
                                  # is_pretokenized=True,
                                  return_special_tokens_mask=True,
                                  return_offsets_mapping=True,
                                  add_special_tokens=True,
                                  return_attention_mask=True,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors="pt"
                                  )
        self.labels = []
        for i in range(len(df.index)):
            row = self.df.iloc[i]
            testo = self.df['Testo'].iloc[i]
            char_labels = list(self.df['Char_segmentation'].iloc[i])
            ends = deque([idx for idx in range(len(char_labels)) if char_labels[idx] == '1'])
            last_token_idx = max(index for index, item in enumerate(self.encodings['special_tokens_mask'][i]) if item == 0)
            encoded_labels = np.ones(len(self.encodings['input_ids'][i]), dtype=int) * -100
            x = ends.popleft()
            for j,e in enumerate(self.encodings['offset_mapping'][i]):
                tok = self.tokenizer.decode(self.encodings['input_ids'][i][j])
                if e[1] != 0:
                    # overwrite label
                    if x >= e[0] and x < e[1]:# Doubt if insert < e[1] because of offset mapping composition
                        encoded_labels[j] = 1
                        if ends: 
                            x = ends.popleft()
                        else:
                            x = -1
                    else:
                        encoded_labels[j] = 0
            encoded_labels[last_token_idx] = 1
            self.labels.append(encoded_labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.df.index)


def find_char_bounds(spans: list, text: str) -> list:
    '''
    Given a list of spans and a text, find the start and end indices of each span in the text.
    Indeces are computed counting CHARS
    
    :param spans: a list of strings to search for
    :type spans: list
    :param text: the text to search
    :type text: str
    :return: A list of tuples, where each tuple contains the start and end index of a span.
    '''
    start = 0
    bounds = []
    last_char = -1
    for span in spans:
        start = text.find(span)
        if start == -1:
            start = last_char + 1
        last_char = start + len(span) - 1
        bounds.append((start, last_char))
        
    return bounds


def find_word_bounds(spans: list, text: str) -> list:
    '''
    Given a list of spans and a text, find the start and end indices of each span in the text.
    Indeces are computed counting WORDS.

    :param spans: a list of strings, each string is a span of text
    :type spans: list
    :param text: the text to be searched
    :type text: str
    :return: A list of tuples, where each tuple is the start and end index of a word in the text.
    '''
    bounds = []
    end = 0
    for span in spans:
        s = span.translate(str.maketrans('', '', string.punctuation))
        word_list = s.split()
        if word_list:   
            text_list = text.translate(str.maketrans('', '', string.punctuation)).split()
            try:
                start = text_list.index(word_list[0], end)
            except:
                if not bounds:
                    start = 0
                else:

                    start = bounds[-1][1] + 1
            end = start + len(word_list) - 1

            bounds.append((start, end))
    return bounds

def find_segmentation(bounds, text):
    """
    It takes a list of tuples representing boundaries and a string of text, and returns a string of 0s and 1s, where 1s indicate
    the end of a span
    
    :param bounds: a list of tuples, each tuple is a span of text
    :param text: the text to be segmented
    :return: A string of 0's and 1's, where 1's indicate the end of a sentence.
    """
    text_list = text.translate(str.maketrans('', '', string.punctuation)).split()
    segmentation = ['0' for i in range(len(text_list))]
    segmentation[-1] = '1'
    
    ends = []
    end = 0
    for span in text_list:
        word_list = span.translate(str.maketrans('', '', string.punctuation)).split()
        try:
            end = text_list.index(word_list[-1], end)
        except:
                end = end + len(word_list) -1
        if end < len(text_list):
            ends.append(end)
    for i in ends:
        segmentation[i] = '1'
    
    return ''.join(segmentation)

def find_segmentation_by_bounds(bounds: list) -> str:
    segmentation = ['0' for i in range(bounds[-1][1] + 1)]
    for bound in bounds:
        if bound[1] < len(segmentation):
            segmentation[bound[1]] = '1'
    segmentation[-1] = '1'
    return ''.join(segmentation)

def clean_text(text:str) -> str:
    #delete \n
    text = text.replace('\n', ' ')
    #text = text.rstrip('\n')
    text = text.rstrip()
    #delete double punctuation
    text =  re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)
    # add space between a word and punctuation
    #text = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)    
    return text

def train_val_split(df, tok_name):
    train_size = 0.8
    train_df = df.sample(frac=train_size)
    val_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)

    return IEHyperionDataset(train_df, tok_name), IEHyperionDataset(val_df, tok_name)
# %%
