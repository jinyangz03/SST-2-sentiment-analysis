# -*- coding: utf-8 -*-
"""
SST-2 Data Processing
Updated from original by YJiangcm:
  - Use tokenizer(...) batch call instead of manual tokenize + convert_tokens_to_ids
  - Supports models without token_type_ids (RoBERTa, DistilBERT) via zero-padding
"""

from torch.utils.data import Dataset
import torch


class DataPrecessForSentence(Dataset):
    """
    Tokenize and encode sentences for BERT-family models.
    Uses the HuggingFace tokenizer's __call__ interface (transformers >= 4.0).
    """

    def __init__(self, bert_tokenizer, df, max_seq_len=50):
        super(DataPrecessForSentence, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.max_seq_len = max_seq_len
        self.input_ids, self.attention_mask, self.token_type_ids, self.labels = self.get_input(df)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx],
        )

    def get_input(self, df):
        sentences = df["s1"].values.tolist()
        labels = df["similarity"].values.tolist()

        # Use tokenizer batch encoding (transformers >= 4.x)
        encoding = self.bert_tokenizer(
            sentences,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        # Some models (RoBERTa, DistilBERT) don't return token_type_ids
        if "token_type_ids" in encoding:
            token_type_ids = encoding["token_type_ids"]
        else:
            token_type_ids = torch.zeros_like(input_ids)

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, labels_tensor
