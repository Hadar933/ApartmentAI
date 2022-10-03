import json
from typing import List, Dict, Tuple
import random

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def load_labels_from_jsonl(jsonl_path='all.jsonl') -> List[str]:
    """
    extracts all the relevant labels from the jsonl file
    :param jsonl_path: str path to the jsonl labels file
    :return: a list of strings that represents the labels
    """
    with open(jsonl_path, 'r', encoding='utf8') as jf:
        jsonL_lst = list(jf)
    all_labels = []
    for jsonL_str in jsonL_lst:
        json_dict = json.loads(jsonL_str)
        item_text = json_dict['text']
        entities_lst = json_dict['entities']
        label_dict = {}
        for entity in entities_lst:
            label_name = entity['label']
            start_offset = entity['start_offset']
            end_offset = entity['end_offset']
            label_dict[label_name] = item_text[start_offset:end_offset]
        string_output = str(label_dict).replace("\'", "")[1:-1]  # removing " ' " and " { ", " } "
        all_labels.append(string_output)
    return all_labels


def train_test_split(labels_arr: List[str],
                     split_ratio: float = 0.75) -> Tuple[List[str], List[str]]:
    """
    splits the labels array to two arrays
    :param labels_arr: a list of labels
    :param split_ratio: the percentage of the data that should be considered train
    :return:
    """
    split_index = int(len(labels_arr) * split_ratio)
    random.shuffle(labels_arr)
    train = labels_arr[:split_index]
    test = labels_arr[split_index:]
    return train, test


def tokenize_a_list(tokenizer: GPT2Tokenizer, lst, max_length) -> List[torch.Tensor]:
    tokens_arr = []
    for item in lst:
        encoder_input = f"{item[:max_length]}<|endoftext|>"
        token = tokenizer.encode(encoder_input)
        tokens_arr.append(torch.tensor(token))
    return tokens_arr


class ApartmentLabels(Dataset):
    def __init__(self, labels_arr: List[str], test_or_train, max_length=1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT")
        self.labels_arr = labels_arr
        self.test_or_train = test_or_train
        self.tokens_lst: List[torch.Tensor] = tokenize_a_list(self.tokenizer, labels_arr, max_length)

    def __getitem__(self, index):
        return self.tokens_lst[index]

    def __len__(self):
        return len(self.tokens_lst)


if __name__ == '__main__':
    labels = load_labels_from_jsonl()
    tr, te = train_test_split(labels)
    x = 2
