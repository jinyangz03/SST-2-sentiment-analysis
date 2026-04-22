# -*- coding: utf-8 -*-
"""
SST-2 Sentiment Analysis Models
- Wrapper classes for different Transformer models
- Unified forward() interface for training and evaluation
"""

import torch
from torch import nn
from transformers import (
    BertForSequenceClassification,
    AlbertForSequenceClassification,
    XLNetForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    AutoTokenizer,
)

# automatically choose GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(BertModel, self).__init__()
        # load pretrained BERT model for classification
        self.bert = BertForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-SST-2", num_labels=2
        )
        # corresponding tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "textattack/bert-base-uncased-SST-2", do_lower_case=True
        )
        self.device = DEVICE

        # control whether to fine-tune model parameters
        for param in self.bert.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        # forward pass through model
        outputs = self.bert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )
        loss = outputs.loss          # training loss
        logits = outputs.logits      # raw predictions
        probabilities = nn.functional.softmax(logits, dim=-1)  # convert to probabilities
        return loss, logits, probabilities


class RobertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(RobertModel, self).__init__()
        # load RoBERTa model
        self.bert = RobertaForSequenceClassification.from_pretrained(
            "textattack/roberta-base-SST-2", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
        self.device = DEVICE

        for param in self.bert.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        # RoBERTa does not use token_type_ids
        outputs = self.bert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class AlbertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(AlbertModel, self).__init__()
        # load ALBERT model
        self.albert = AlbertForSequenceClassification.from_pretrained(
            "textattack/albert-base-v2-SST-2", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "textattack/albert-base-v2-SST-2", do_lower_case=True
        )
        self.device = DEVICE

        for param in self.albert.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.albert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class XlnetModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(XlnetModel, self).__init__()
        # load XLNet model
        self.xlnet = XLNetForSequenceClassification.from_pretrained(
            "textattack/xlnet-base-cased-SST-2", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained("textattack/xlnet-base-cased-SST-2")
        self.device = DEVICE

        for param in self.xlnet.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.xlnet(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class DistilBertModel(nn.Module):
    """
    DistilBERT: smaller and faster version of BERT
    Does NOT use token_type_ids
    """

    def __init__(self, requires_grad=True):
        super(DistilBertModel, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english", do_lower_case=True
        )
        self.device = DEVICE

        for param in self.distilbert.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        # ignore token_type_ids
        outputs = self.distilbert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities