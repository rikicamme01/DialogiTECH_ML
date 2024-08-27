import time
import datetime
from torch.nn import utils

import torchmetrics
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup
from nltk.metrics.segmentation import windowdiff, ghd, pk
import numpy as np

from utils.utils import format_time
from datasetss.ie_hyperion_dataset import find_segmentation_by_bounds, find_word_bounds
from models.bert_segmenter import decode_segmentation, split_by_prediction, extract_active_preds

# This class is a wrapper for the training and testing of a Bert model for text segmentation
class BertSegTrainer():

    def fit(self, model, train_dataset, val_dataset, batch_size, lr, n_epochs, loss_fn):

        output_dict = {}
        output_dict['train_metrics'] = []
        output_dict['train_loss'] = []
        output_dict['val_metrics'] = []
        output_dict['val_loss'] = []

        torch.cuda.empty_cache()
        # ----------TRAINING

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        # Creation of Pytorch DataLoaders with shuffle=True for the traing phase
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True)

        # Adam algorithm optimized for tranfor architectures
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        #scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=300)

        # Scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()

        # Setup for training with gpu
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        loss_fn.to(device)

        # For each epoch...
        for epoch_i in range(0, n_epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, n_epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode: Dropout layers are active
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # Progress update every 40 batches.
                if step % 10 == 0 and not step == 0:
                    # Compute time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                        step, len(train_dataloader), elapsed))

                # Unpack this training batch from the dataloader.
                #
                #  copy each tensor to the GPU using the 'to()' method
                #
                # 'batch' contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch['input_ids'].to(device)
                b_input_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)

                # clear any previously calculated gradients before performing a
                # backward pass
                model.zero_grad()

                # Perform a forward pass in mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

                    #loss = outputs[0]
                    
                    logits = outputs[1]

                    loss = loss_fn(logits.view(-1, model.num_labels), b_labels.view(-1))

                # Move logits and labels to CPU
                logits = logits.detach().cpu()

                # Perform a backward pass to compute the gradients in MIXED precision
                scaler.scale(loss).backward()

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end.
                total_train_loss += loss.item()

                # Unscales the gradients of optimizer's assigned params in-place before the gradient clipping
                scaler.unscale_(optimizer)

                # Clip the norm of the gradients to 1.0.
                # This helps and prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient in MIXED precision
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()

            # Compute the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)

            output_dict['train_loss'].append(avg_train_loss)

            # Measure how long this epoch took.
            training_time = format_time(time.time() - t0)

            print("")
            print("  Average training loss: {0:.3f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(training_time))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure performance on
            # the validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode: the dropout layers behave differently
            model.eval()

            total_val_loss = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:

                # Unpack this training batch from our dataloader.
                #
                # copy each tensor to the GPU using the 'to()' method
                #
                # 'batch' contains three pytorch tensors:
                #   [0]: input ids
                #   [1]: attention masks
                #   [2]: labels
                b_input_ids = batch['input_ids'].to(device)
                b_input_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for training.
                with torch.no_grad():

                    # Forward pass, calculate logits
                    # argmax(logits) = argmax(Softmax(logits))
                    outputs = model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                    #loss = outputs[0]
                    
                    logits = outputs[1]

                    loss = loss_fn(logits.view(-1, model.num_labels), b_labels.view(-1))

                # Accumulate the validation loss.
                total_val_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu()

            print('VALIDATION: ')

            # Compute the average loss over all of the batches.
            avg_val_loss = total_val_loss / len(validation_dataloader)
            output_dict['val_loss'].append(avg_val_loss)

            # Measure how long the validation run took.
            validation_time = format_time(time.time() - t0)

            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))

        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(
            format_time(time.time()-total_t0)))
        
        return output_dict

    def test(self, model, test_dataset, batch_size, loss_fn):
        # ========================================
        #               Test
        # ========================================
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        
        output_dict = {}

        # Setup for testing with gpu
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        loss_fn.to(device)

        print("")
        print("Running Test...")
        t0 = time.time()

        full_labels = []

        model.eval()

        total_test_loss = 0

        # Evaluate data for one epoch
        for batch in test_dataloader:
            b_input_ids = batch['input_ids'].to(device)
            b_input_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            with torch.no_grad():

                # Forward pass, calculate logits
                # argmax(logits) = argmax(Softmax(logits))
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                #loss = outputs[0]
                
                logits = outputs[1]
                loss = loss_fn(logits.view(-1, model.num_labels), b_labels.view(-1))

            # Accumulate the test loss.
            total_test_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu()  # shape (batch_size, seq_len, num_labels
            full_probs = logits.softmax(dim=-1)
            full_probs = full_probs.tolist()
            full_labels += [decode_segmentation(p, 0.5) for p in full_probs]

        active_labels = [extract_active_preds(full_labels[i], test_dataset[i]['special_tokens_mask'].tolist()) for i in range(len(full_labels))]
        pred_spans = [split_by_prediction(e, test_dataset[i]['input_ids'].tolist(), test_dataset[i]['offset_mapping'].tolist(), test_dataset.df.iloc[i]['Testo'], test_dataset.tokenizer) for i,e in enumerate(active_labels)]
        pred_word_bounds = [find_word_bounds(e, test_dataset.df.iloc[i]['Testo']) for i,e in enumerate(pred_spans)]
        norm_pred_word_bounds = [normalize_bounds_by_repertoire(e, test_dataset.df.iloc[i]) for i,e in enumerate(pred_word_bounds)]   

        avg_test_loss = total_test_loss / len(test_dataloader)
        test_time = format_time(time.time() - t0)

        #output_dict['pred'] = labels
        output_dict['loss'] = avg_test_loss
        output_dict['bounds'] = pred_word_bounds
        output_dict['metrics'] = compute_metrics(pred_word_bounds, test_dataset)
        output_dict['normalized_metrics'] = compute_metrics(norm_pred_word_bounds, test_dataset)

        count = 0
        for pred in pred_spans:
            count += len(pred) 
        output_dict['predicted_spans'] = count

        print("  Test Loss: {0:.2f}".format(avg_test_loss))
        print("  Test took: {:}".format(test_time))

        return output_dict

# computes 'windowdiff' 'ghd' 'pk'  'iou'
def compute_metrics(preds, dataset):
    met_list = []
    for i in range(len(dataset.df.index)):
        if len(dataset.df['Segmentation'].iloc[i]) >= 20:
            seg_pred = find_segmentation_by_bounds(preds[i])
            seg_pred = seg_pred[:len(dataset.df['Segmentation'].iloc[i])]
            seg_gt_trunk = dataset.df['Segmentation'].iloc[i][:len(seg_pred)] # manages predictiones in text with n_tokens  > 512

            wd_value = windowdiff(seg_gt_trunk, seg_pred,  20)
            ghd_value = ghd(seg_gt_trunk, seg_pred,20, 20, 1)
            pk_value = pk(seg_gt_trunk, seg_pred, 20)

            text_IoUs = []
            for bound in preds[i]:
                IoUs = compute_IoUs(bound, dataset.df['Bounds'].iloc[i])
                best = np.argmax(IoUs)
                text_IoUs.append(IoUs[best])

            met_dict = {
                'windowdiff' : wd_value,
                'ghd' : ghd_value,
                'pk' : pk_value,
                'iou' : text_IoUs
                }
            met_list.append(met_dict)
    IoUs = [e['iou'] for e in met_list]
    flat_IoUs = [item for sublist in IoUs for item in sublist]
    out = {
            'windowdiff' : np.mean([e['windowdiff'] for e in met_list]),
            'ghd' : np.mean([e['ghd'] for e in met_list]),
            'pk' : np.mean([e['pk'] for e in met_list]),
            'iou' : np.mean(flat_IoUs)
            }
    return out


def IoU(A, B):
    """
    Computes Intersection over Union
    
    :param A: The first interval
    :param B: The ground truth bounding box
    :return: The intersection over union of two intervals.
    """
    if A == B:
        return 1
    start = max(A[0], B[0])
    end = min(A[1], B[1])
    if(start > end):
        return 0
    intersection = end - start
    return intersection / (A[1] - A[0] + B[1] - B[0] - intersection)

def compute_IoUs(pred_bounds, gt_spans):
    """
    Given a list of predicted spans and a list of ground truth spans, 
    compute the IoU between each pair of spans
    
    :param pred_bounds: a tuple of (start, end) denoting the predicted answer
    :param gt_spans: a list of tuples of the form (start, end) representing the spans of each ground
    truth annotation
    :return: a list of IoUs for each ground truth span.
    """
    IoUs = []
    for gt_bounds in gt_spans:
        IoUs.append(IoU(pred_bounds, gt_bounds)) 
    return IoUs


def intersection(A, B):
    """
    Compoute intersection between two span boundaries represented as tuples (start, end)
    """
    if A == B:
        return 1
    start = max(A[0], B[0])
    end = min(A[1], B[1])
    if(start > end):
        return 0
    return end - start + 1

def normalize_bounds_by_repertoire(bounds, sample):
    """
    For each bound in the list of bounds, find the ground truth bound that it has the most overlap
    with, and then group all bounds that have the same ground truth bound together
    
    :param bounds: list of tuples, each tuple is a bounding box
    :param sample: a dictionary with the following keys:
    :return: A list of tuples, where each tuple is a span of text.
    """
    bounds_w_rep = []
    for bound in bounds:
        intersections = []
        for gt_bound in sample['Bounds']:
            intersections.append(intersection(bound, gt_bound))
        rep_idx = np.argmax(intersections)
        bounds_w_rep.append({
            'Bounds': bound,
            'Repertorio': sample['Repertori'][rep_idx]
            })
    normalized = []
    for i in range(len(bounds_w_rep)):
        #normalized is not empty
        if normalized:
            if normalized[-1]['Repertorio'] == bounds_w_rep[i]['Repertorio']:
                new_span = (normalized[-1]['Bounds'][0], bounds_w_rep[i]['Bounds'][1])
                new_span_features = {
                    'Bounds' : new_span, 
                    'Repertorio' : bounds_w_rep[i]['Repertorio']
                    }
                del normalized[-1]
                normalized.append(new_span_features)
            else:
                normalized.append(bounds_w_rep[i])
        else:
            normalized.append(bounds_w_rep[i])
    return [e['Bounds'] for e in normalized]