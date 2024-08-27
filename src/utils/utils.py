#%%
import random
import os

import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


#Deterministic mode
def seed_everything(seed=1464):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def plot_loss(loss, val_loss):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xticks(range(1,len(loss)+1))
    plt.plot(range(1,len(loss)+1), loss, label='train')
    plt.plot(range(1,len(val_loss)+1), val_loss, label='val')
    plt.title('loss')
    plt.legend()
    #plt.savefig('loss.png')
    return fig

def plot_f1(f1, val_f1):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.xticks(range(1,len(f1)+1))
    plt.plot(range(1,len(f1)+1), f1, label='train')
    plt.plot(range(1,len(val_f1)+1), val_f1, label='val')
    plt.title('f1')
    plt.legend()
    #plt.savefig('f1.png')
    return fig

def plot_confusion_matrix(y_true, pred, labels):
    fig, ax = plt.subplots(figsize=(20, 20))
    disp = ConfusionMatrixDisplay.from_predictions(y_true, pred, display_labels=labels, normalize='true', values_format='.2f')
    disp.plot(cmap="Blues", values_format='.2g',xticks_rotation='vertical', ax=ax)
    return disp.figure_
# %%
