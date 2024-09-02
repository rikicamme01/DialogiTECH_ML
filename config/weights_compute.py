#%% 
import pandas as pd
import numpy as np

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

df = pd.read_excel('../data/classifier/dataset_train.xlsx')
df['Repertorio'] = df['Repertorio'].str.lower()
N= df.shape[0]
k = 24
weight_list = []
#%%   N/24*c_i
for rep in LABELS:
    num_rep = (df['Repertorio'] == rep).sum()
    weight = N/(k*num_rep)
    
    print(f'loss_weight ({rep}):{weight}')
    weight_list.append(weight)

print(weight_list)


# %%   1/sqrt(c_i)
for rep in LABELS:
    num_rep = (df['Repertorio'] == rep).sum()
    weight = 1/np.sqrt(num_rep)
    
    print(f'loss_weight ({rep}):{weight}')
    weight_list.append(weight)

print(weight_list)

# %% C_max/c_i
C_max = 0
for rep in LABELS:
    num_rep = (df['Repertorio'] == rep).sum()
    if num_rep > C_max:
        C_max = num_rep

for rep in LABELS:
    num_rep = (df['Repertorio'] == rep).sum()
    weight = C_max/num_rep
    
    print(f'loss_weight ({rep}):{weight}')
    weight_list.append(weight)

print(weight_list)
# %%
