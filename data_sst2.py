# -*- coding: utf-8 -*-
"""
SST-2 Data Processing
- Use tokenizer batch encoding instead of manual tokenization
- Compatible with models without token_type_ids (e.g., RoBERTa)
"""

from torch.utils.data import Dataset
import torch


class DataPrecessForSentence(Dataset):
    """
    Custom Dataset for sentence classification (SST-2 style).
    Converts raw text into model inputs.
    """

    def __init__(self, bert_tokenizer, df, max_seq_len=50):
        super(DataPrecessForSentence, self).__init__()
        self.bert_tokenizer = bert_tokenizer  # HuggingFace tokenizer
        self.max_seq_len = max_seq_len        # max sequence length
        # preprocess and store all tensors
        self.input_ids, self.attention_mask, self.token_type_ids, self.labels = self.get_input(df)

    def __len__(self):
        # return dataset size
        return len(self.labels)

    def __getitem__(self, idx):
        # return one sample (used by DataLoader)
        return (
            self.input_ids[idx],
            self.attention_mask[idx],
            self.token_type_ids[idx],
            self.labels[idx],
        )

    def get_input(self, df):
        # extract sentences and labels from dataframe
        sentences = df["s1"].values.tolist()
        labels = df["similarity"].values.tolist()

        # batch tokenize sentences
        encoding = self.bert_tokenizer(
            sentences,
            max_length=self.max_seq_len,
            padding="max_length",   # pad shorter sentences
            truncation=True,        # truncate longer sentences
            return_tensors="pt",    # return PyTorch tensors
        )

        input_ids = encoding["input_ids"]              # token ids
        attention_mask = encoding["attention_mask"]    # mask for padding

        # handle models without token_type_ids (e.g., RoBERTa)
        if "token_type_ids" in encoding:
            token_type_ids = encoding["token_type_ids"]
        else:
            token_type_ids = torch.zeros_like(input_ids)

        # convert labels to tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return input_ids, attention_mask, token_type_ids, labels_tensor