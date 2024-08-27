#%% 
import pandas as pd

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

df = pd.read_excel('../data/classifier/dataset_finale_train.xlsx')
df['Repertorio'] = df['Repertorio'].str.lower()
N= df.shape[0]
k = 24
weight_list = []

for rep in LABELS:
    num_rep = (df['Repertorio'] == rep).sum()
    weight = N/(k*num_rep)
    
    print(f'loss_weight ({rep}):{weight}')
    weight_list.append(weight)

print(weight_list)
# %%
