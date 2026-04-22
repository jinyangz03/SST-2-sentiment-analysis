# -*- coding: utf-8 -*-
"""
Training / Validation / Test utilities
- Provide training, validation and testing loops
- Includes metric computation and performance tracking
"""

import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


def Metric(y_true, y_pred):
    """Compute and print classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="macro")

    target_names = ["negative", "positive"]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    # print summary metrics
    print(
        "Accuracy: {:.4f}\nPrecision: {:.4f}\nRecall: {:.4f}\nF1: {:.4f}".format(
            accuracy, macro_precision, macro_recall, weighted_f1
        )
    )
    print("classification_report:\n")
    print(report)


def correct_predictions(output_probabilities, targets):
    """Count number of correct predictions in a batch."""
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def train(model, dataloader, optimizer, epoch_number, max_gradient_norm):
    """Train model for one epoch."""
    model.train()
    device = model.device

    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0

    # iterate through training batches
    tqdm_iter = tqdm(dataloader, desc="Training")
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(tqdm_iter):
        batch_start = time.time()

        # move data to device (CPU/GPU)
        seqs = batch_seqs.to(device)
        masks = batch_seq_masks.to(device)
        segments = batch_seq_segments.to(device)
        labels = batch_labels.to(device)

        optimizer.zero_grad()

        # forward + backward pass
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()

        # gradient clipping to avoid exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

        optimizer.step()

        # accumulate statistics
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)

        # update progress bar
        tqdm_iter.set_description(
            "Epoch {} | avg batch time: {:.3f}s | loss: {:.4f}".format(
                epoch_number,
                batch_time_avg / (batch_index + 1),
                running_loss / (batch_index + 1),
            )
        )

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)

    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader):
    """Evaluate model on validation set."""
    model.eval()
    device = model.device

    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []

    # disable gradient computation
    with torch.no_grad():
        for batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels in dataloader:
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)

            loss, logits, probabilities = model(seqs, masks, segments, labels)

            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)

            # store probabilities for AUC calculation
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / len(dataloader.dataset)

    # compute ROC-AUC score
    auc = roc_auc_score(all_labels, all_prob)

    return epoch_time, epoch_loss, epoch_accuracy, auc, all_prob


def test(model, dataloader):
    """Test model and return predictions and metrics."""
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels in dataloader:
            batch_start = time.time()

            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)

            # forward pass (no loss needed here)
            _, logits, probabilities = model(seqs, masks, segments, labels)

            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start

            # store outputs
            all_prob.extend(probabilities[:, 1].cpu().numpy())
            _, preds = probabilities.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= len(dataloader.dataset)

    return batch_time, total_time, accuracy, all_prob, all_preds, all_labels