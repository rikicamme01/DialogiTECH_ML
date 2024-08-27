#%%
import yaml
import os
import sys 
import time

sys.path.append(os.path.dirname(sys.path[0]))
#sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import torch
import neptune
run = neptune.init_run(
    project="riccardocamellini01/DialogiTECH",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTkzYjg4Yy1jMTVhLTQ2NDktYmFjOC0yNzZmZDEyMDFlOTcifQ==",
) 

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from datasetss.hyperion_dataset import HyperionDataset
from datasetss.hyperion_dataset import train_val_split
from trainers.bert_cls_trainer import BertClsTrainer
from utils.utils import seed_everything
from utils.utils import plot_confusion_matrix, plot_f1, plot_loss

#%%
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Costruisci il percorso relativo del file da aprire
try: 
    with open (script_dir + '/config/bert_cls_train.yml', 'r') as file:
        config = yaml.safe_load(file)        
except Exception as e:
    print('Error reading the config file')
    sys.exit(1)
print(f"config file loaded and save is[{config['save']}]")

#%%
seed_everything(config['seed'])

df = pd.read_excel(script_dir + '/data/classifier/dataset_finale_train.xlsx', na_filter=False)
test_df = pd.read_csv(script_dir + '/data/classifier/hyperion_test.csv', na_filter=False)

run['config'] = config


model_name = config['model']

train_dataset, val_dataset = train_val_split(df, model_name, subsample=False)
test_dataset = HyperionDataset(test_df, model_name)

trainer = BertClsTrainer()


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=24) #
model.name = model_name

#%%
start_time = time.time()
history = trainer.fit(model,
            train_dataset, 
            val_dataset,
            config['batch_size'],
            float(config['learning_rate']),
            config['n_epochs'],
            torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights'])))

run['history'] = history

out = trainer.test(model,test_dataset, config['batch_size'], torch.nn.CrossEntropyLoss(weight = torch.Tensor(config['loss_weights'])))
run['test/metrics'] = out['metrics']
run['test/loss'] = out['loss']

cm = plot_confusion_matrix(out['gt'], out['pred'], test_dataset.labels_list())
run["confusion_matrix"].upload(neptune.types.File.as_image(cm))

fig = plot_loss(history['train_loss'], history['val_loss'])
run["loss_plot"].upload(neptune.types.File.as_image(fig))


#%%
#hf_token = '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' # token modello Michele
hf_token = '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'
if config['save']:
    model.push_to_hub("BERT_DialogicaPD", use_temp_dir=True, token=hf_token)
    AutoTokenizer.from_pretrained(model_name).push_to_hub("BERT_DialogicaPD", use_temp_dir=True, token=hf_token)
    print('Model correctly push to hub')
else:
    print("The 'save' field in config file has to be True")
# %%
