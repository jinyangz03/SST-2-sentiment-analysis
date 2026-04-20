# -*- coding: utf-8 -*-
"""
SST-2 Sentiment Analysis Models
Updated from original by YJiangcm:
  - Compatible with PyTorch 2.x and transformers 4.x
  - Fixed forward() return unpacking (now uses .loss / .logits attributes)
  - Added DistilBertModel
  - device handling uses torch.device("cuda" if available else "cpu")
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            "textattack/bert-base-uncased-SST-2", num_labels=2
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "textattack/bert-base-uncased-SST-2", do_lower_case=True
        )
        self.requires_grad = requires_grad
        self.device = DEVICE
        for param in self.bert.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        outputs = self.bert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            token_type_ids=batch_seq_segments,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities


class RobertModel(nn.Module):
    def __init__(self, requires_grad=True):
        super(RobertModel, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2", num_labels=2)
        
        self.tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
        self.requires_grad = requires_grad
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
        self.albert = AlbertForSequenceClassification.from_pretrained(
            "albert-base-v2", num_labels=2  # use base for speed; original used xxlarge
        )
        self.tokenizer = AutoTokenizer.from_pretrained("albert-base-v2", do_lower_case=True)
        self.requires_grad = requires_grad
        self.device = DEVICE
        for param in self.albert.parameters():
            param.requires_grad = True

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
        self.xlnet = XLNetForSequenceClassification.from_pretrained(
            "xlnet-base-cased", num_labels=2  # base for accessibility
        )
        self.tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
        self.requires_grad = requires_grad
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
    NEW MODEL: DistilBERT for sequence classification.
    DistilBERT is a smaller, faster distilled version of BERT.
    It does NOT use token_type_ids.
    """

    def __init__(self, requires_grad=True):
        super(DistilBertModel, self).__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", do_lower_case=True)
        
        self.requires_grad = requires_grad
        self.device = DEVICE
        for param in self.distilbert.parameters():
            param.requires_grad = requires_grad

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        # DistilBERT does not use token_type_ids; ignore batch_seq_segments
        outputs = self.distilbert(
            input_ids=batch_seqs,
            attention_mask=batch_seq_masks,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
