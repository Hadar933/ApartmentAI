import json
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


def load_data_from_jsonl(jsonl_path='all.jsonl') -> Tuple[List[str], List[str]]:
    """
    extracts all the relevant text and labels from the jsonl file
    :param jsonl_path: str path to the jsonl labels file
    :return: two lists of text and label text
    """
    with open(jsonl_path, 'r', encoding='utf8') as jf:
        jsonL_lst = list(jf)
    data = []
    labels = []
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
        data.append(item_text)
        labels.append(string_output)
    return data, labels


def train_test_split(data_arr: List[str], labels_arr: List[str],
                     split_ratio: float = 0.75) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    splits the labels array to two arrays
    :param data_arr: a list of texts
    :param labels_arr: a list of labels
    :param split_ratio: the percentage of the data that should be considered train
    :return:
    """
    split_index = int(len(labels_arr) * split_ratio)
    Xy = np.array([data_arr, labels_arr])
    np.random.shuffle(np.transpose(Xy))  # shuffling the columns so the order of data-label is preserved
    X_train, X_test = Xy[0, :split_index], Xy[0, split_index:]
    y_train, y_test = Xy[1, :split_index], Xy[1, split_index:]
    return X_train, y_train, X_test, y_test


def _tokenize_a_list(tokenizer: GPT2Tokenizer, lst, max_length) -> List[torch.Tensor]:
    tokens_arr = []
    for item in lst:
        encoder_input = f"{item[:max_length]}<|endoftext|>"
        token = tokenizer.encode(encoder_input)
        tokens_arr.append(torch.tensor(token))
    return tokens_arr


class ApartmentDataset(Dataset):
    def __init__(self, data_arr: np.ndarray, labels_arr: np.ndarray, is_train: bool, max_length: int = 1024):
        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT")
        self.X = data_arr
        self.y = labels_arr
        self.is_train = is_train
        self.X_tokenized: List[torch.Tensor] = _tokenize_a_list(self.tokenizer, self.X, max_length)
        self.y_tokenized: List[torch.Tensor] = _tokenize_a_list(self.tokenizer, self.y, max_length)

    def __getitem__(self, index):
        return self.X_tokenized[index], self.y_tokenized[index]

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':
    X, y = load_data_from_jsonl()
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    train_ds = ApartmentDataset(X_train, y_train, True)
    test_ds = ApartmentDataset(X_test, y_test, False)
