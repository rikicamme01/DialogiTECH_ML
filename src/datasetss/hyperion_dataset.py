import re

import pandas as pd
import torch
from sklearn import preprocessing
from transformers import AutoTokenizer

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

LABELS = [
                'anticipazione',
                'causa',
                'commento',
                'conferma',
                'considerazione',
                'contrapposizione',
                'deresponsabilizzazione',
                'descrizione',
                'dichiarazione di intenti',
                'generalizzazione',
                'giudizio',
                'giustificazione',
                'implicazioni',
                'non risposta',
                'opinione',
                'possibilita',
                'prescrizione',
                'previsione',
                'proposta',
                'ridimensionamento',
                'sancire',
                'specificazione',
                'valutazione',
                'riferimento obiettivo',
        ]
MIN_COLS_ALLOWED = [
                'Testo',
                'Stralcio',
                'Repertorio',
                'Alternative',
                'Età',
                'Genere',
                'Ruolo',
]
COLS_ALLOWED_PLOTTER =[
                    'num_risposta',
                    'num_domanda', #-> da ricordarsi quando si creerà file excel con risposte forms
                    'Età','età','Eta','ETÀ',
                    'Genere', 'GENERE', 'genere', 'Sesso', 'sesso', 'SESSO',
                    'Ruolo', 'RUOLO', 'ruolo',
                    'Repertorio','REPERTORIO', 'repertorio',
                    'Arcipelago', 'ARCIPELAGO', 'arcipelago','Ads', 
]
REP_GENERATIVI = ['descrizione','proposta','considerazione','anticipazione','riferimento obiettivo']
REP_IBRIDI = ['possibilita', 'valutazione', 'dichiarazione di intenti', 'specificazione', 'conferma', 'prescrizione', 'ridimensionamento', 'implicazioni']
REP_MANTENIMENTO = ['sancire', 'causa', 'previsione', 'giudizio', 'opinione', 'deresponsabilizzazione', 'giustificazione', 'contrapposizione', 'generalizzazione', 'commento', 'non risposta']

class HyperionDataset(torch.utils.data.Dataset):
    

    def __init__(self, df, tokenizer_name):
        #fill_null_features(df)
        df = filter_empty_labels(df)
        #df = twitter_preprocess(df)
        df = to_lower_case(df)
        uniform_labels(df)          
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.encodings = tokenizer(
        df['Stralcio'].tolist(),
        max_length=512,
        add_special_tokens=True,
        return_attention_mask=True,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
        self.labels = encode_labels(df).tolist()    

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
    def labels_list(self):
        return LABELS


# Dataset loading and preprocessing
def fill_null_features(df):
    """
    For each column in the list ['Domanda','Testo'], if the value of the cell is null, then replace it
    with the value of the previous cell to obtain a full dataset 
    
    :param df: the dataframe
    """
    for c in ['Domanda','Testo']:
        for i in range(0,len(df.index)):  
            if not df[c][i]:
                j=i
                while j>0: 
                    j-=1
                    if df[c][j]:
                        df[c][i] = df[c][j]
                        break


#Delete examples with empty label
def filter_empty_labels(df):
    filter = df["Repertorio"] != ""
    return df[filter]

#Convert to lower case
def to_lower_case(df):
    return df.map(str.lower)


#Lables uniformation uncased
def uniform_labels(df):
    df['Repertorio'].replace('implicazione','implicazioni', inplace=True)
    df['Repertorio'].replace('previsioni','previsione', inplace=True)


def encode_labels(df):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.transform(df['Repertorio'])

def encode_str_label(rep:str):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.transform([rep])

def decode_labels(encoded_labels):
    le = preprocessing.LabelEncoder()
    le.fit(LABELS)
    return le.inverse_transform(encoded_labels)

'''probabilities is a tensor converted into a list of n elements, one for each stralcio predicted. 
    probabilities[i] = is a list of 23 probabilities for each repertorio

    dict_per_stralci is an array of dictionaries, one for each stralcio
 '''

def decode_labels_vector(probabilities):
    dict_per_stralci = []
    for prob in probabilities:
        if len(prob) != len(LABELS):
            raise ValueError(f"Le liste delle chiavi e dei valori devono avere la stessa lunghezza: {len(prob)},{len(LABELS)}")
        dict_per_stralci.append(dict(zip(LABELS, prob)))  
    return dict_per_stralci

def twitter_preprocess(text:str) -> str:
    """
    - It takes a string as input
    - It returns a string as output
    - It does the following:
        - Normalizes terms like url, email, percent, money, phone, user, time, date, number
        - Annotates hashtags
        - Fixes HTML tokens
        - Performs word segmentation on hashtags
        - Tokenizes the string
        - Replaces tokens extracted from the text with other expressions
        - Removes non-alphanumeric characters
        - Removes extra whitespaces
        - Removes repeated characters
        - Removes leading and trailing whitespaces
    
    :param text: The text to be processed
    :type text: str
    :return: A string with the preprocessed text.
    """
    text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag"},
    fix_html=True,  # fix HTML tokens
    
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
    )

    text = str(" ".join(text_processor.pre_process_doc(text)))
    text = re.sub(r"[^a-zA-ZÀ-ú</>!?♥♡\s\U00010000-\U0010ffff]", ' ', text)
    text = re.sub(r"\s+", ' ', text)
    text = re.sub(r'(\w)\1{2,}',r'\1\1', text)
    text = re.sub ( r'^\s' , '' , text )
    text = re.sub ( r'\s$' , '' , text )

    return text
    

def train_val_split(df, tok_name,  val_perc=0.2, subsample = False):
    """
    It takes a dataframe, a tokenizer name, a validation percentage and a subsample flag. It then splits
    the dataframe into a training and validation set, and returns a HyperionDataset object for each
    
    :param df: the dataframe containing the data
    :param tok_name: the name of the tokenizer to use
    :param val_perc: the percentage of the data that will be used for validation
    :param subsample: if True, subsample the dataset to 50 samples per class, defaults to False
    (optional)
    :return: A tuple of two datasets, one for training and one for validation.
    """
    gb = df.groupby('Repertorio')
    train_list = []
    val_list = []
    for x in gb.groups:
        if subsample:
            class_df = gb.get_group(x).head(50)
        else:
            class_df = gb.get_group(x)

        # Validation set creation
        val = class_df.sample(frac=val_perc)
        train = pd.concat([class_df,val]).drop_duplicates(keep=False)

        #train_list.append(train.head(500))
        train_list.append(train)
        val_list.append(val)

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    return HyperionDataset(train_df, tok_name), HyperionDataset(val_df, tok_name)