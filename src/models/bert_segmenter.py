
from collections import deque
from typing import List

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

import pandas as pd 


# The class takes a string as input, tokenizes it, feeds it to the model, and returns a list of
# strings, where each string is a segment
class BertSegmenter():
    def __init__(self):
        self.model = AutoModelForTokenClassification.from_pretrained('MiBo/SegBert')
        self.tokenizer = AutoTokenizer.from_pretrained('MiBo/SegBert')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text:str) -> List[str]:
        encoded_text = self.tokenizer(text,
                                    return_special_tokens_mask=True,
                                    return_offsets_mapping=True,
                                    add_special_tokens=True,
                                    return_attention_mask=True,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors = 'pt'
                                    )
        
        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        logits = self.model(input_ids, attention_mask)['logits']
        
        full_probs = logits.softmax(dim=-1)
        full_probs = full_probs.view(full_probs.size(1), full_probs.size(2))

        full_labels = decode_segmentation(full_probs, 0.5)
        active_labels = extract_active_preds(full_labels, encoded_text['special_tokens_mask'][0].tolist())

        x = split_by_prediction(active_labels, encoded_text['input_ids'][0].tolist(), encoded_text['offset_mapping'][0].tolist(), text, self.tokenizer)
        return x

def extract_active_preds(preds:list, special_tokens:list) -> deque:
    """
    It takes a list of predictions and a list of special tokens and returns a list of active
    predictions deleting th ones corresponding to special tokens
    
    :param preds: list of predictions
    :type preds: list
    :param special_tokens: list of special tokens
    :type special_tokens: list
    :return: The active predictions are being returned.
    """
    active = []
    for i, e in enumerate(special_tokens):
        if(e == 0):
            active.append(preds[i])
    active[-1] = 1
    return active
         

def decode_segmentation(probs, threshold):  #one sample
    """
    It takes a list of probabilities and a threshold, and returns a list of 0s and 1s, where 1s are
    where the probability is greater than the threshold
    
    :param probs: a list of probabilities for each word in the sentence
    :param threshold: the threshold for the probability of a word being a keyword
    :return: A list of 1s and 0s.
    """
    if threshold < 0 or threshold > 1:
        return None
    segmentation = []
    for prob in probs:
        if prob[1] >= threshold:
            segmentation.append(1)
        else:
            segmentation.append(0)
    segmentation[-1] = 1
    return segmentation

def split_by_prediction(pred:list, input_ids:list, offset_mapping:list, text:str, tokenizer) -> list:
    """
    Split the text propagating the boundary to the last subword token
    
    :param pred: list of predictions
    :type pred: list
    :param input_ids: the tokenized input text
    :type input_ids: list
    :param offset_mapping: a list of tuples, where each tuple is (start_offset, end_offset)
    :type offset_mapping: list
    :param text: the text to be split
    :type text: str
    :param tokenizer: the tokenizer used to tokenize the text
    :return: a list of strings, where each string is a span of text.
    """
    subword_flags = [False for i in range(len(input_ids))]
    for i in  range(len(input_ids)):
        if offset_mapping[i][1] != 0:
            if tokenizer.decode(input_ids[i])[:2] == '##':
                subword_flags[i] = True
    
    for i in range(len(pred)-1):
        if pred[i] == 1 and subword_flags[i]:
                pred[i] = 0
                pred[i + 1] = 1

    spans = []
    start = 0
    j=0
    for i in range(len(offset_mapping)):
        if offset_mapping[i][1] != 0:
            x = pred[j]
            j += 1
            if x == 1:
                spans.append(text[start:offset_mapping[i][1]])
                start = offset_mapping[i][1]
    if not spans:
        spans.append(text)
    return spans

def normalize(bounds:list, reps:list):
    """
    It takes a list of boundaries and if adjacent segments corresponds to the same repertoires the segments are joined
    
    :param bounds: list of tuples, each tuple is a start and end of a segment
    :type bounds: list
    :param reps: list of the number of repetitions of each interval
    :type reps: list
    :return: A series of two lists. The first list is a list of tuples, where each tuple is a start and
    end point of a range, and the second ais the list of repertoires
    """
    norm_bounds = []
    norm_reps = []
    
    for i in range(len(bounds)):
        if norm_reps and norm_reps[-1] == reps[i]:
            norm_bounds[-1] = (norm_bounds[-1][0], bounds[i][1])
        else:
            norm_bounds.append(bounds[i])
            norm_reps.append(reps[i])
    return pd.Series([norm_bounds, norm_reps])

    