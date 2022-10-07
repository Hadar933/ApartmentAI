import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import json
import random
from typing import List, Dict, Tuple

PYTORCH_TOKEN = "pt"  # use this with tokenizer to return a pytorch tensor


def _load_data_from_jsonl(jsonl_path='all.jsonl') -> List[str]:
    """
    extracts all the relevant text and labels from the jsonl file
    :param jsonl_path: str path to the jsonl labels file
    :return: a list of data text + label text
    """
    with open(jsonl_path, 'r', encoding='utf8') as jf:
        jsonL_lst = list(jf)
    data = []
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
        desired_output = str(label_dict).replace("\'", "")[1:-1]  # removing " ' " and " { ", " } "
        concatenated_data = f"INPUT: {item_text}.\n OUTPUT: {desired_output}"
        data.append(concatenated_data)
    return data


def _train_test_split(data_arr: List[str], split_ratio: float = 0.75) -> Tuple[List[str], List[str]]:
    """
    splits the labels array to two arrays
    :param data_arr: a list of the text itself + the desired outputs
    :param split_ratio: the percentage of the data that should be considered train
    :return: two lists of the splitted data
    """
    random.shuffle(data_arr)
    split_index = int(len(data_arr) * split_ratio)
    train, test = data_arr[:split_index], data_arr[split_index:]
    return train, test


def _tokenize_data_and_labels(tokenizer: GPT2Tokenizer, data: List[str]) -> Dict[str, torch.Tensor]:
    """

    :param tokenizer:
    :param data:
    :return:
    """
    # generating a token for the word "OUTPUT"
    output_token = tokenizer(' OUTPUT', return_tensors=PYTORCH_TOKEN)['input_ids'][0][0]

    # tokenizing all the provided data
    encoded_data = tokenizer(data, padding=True, truncation=True, max_length=1024, return_tensors=PYTORCH_TOKEN,
                             return_attention_mask=True)

    # extracting the location of the token "OUTPUT" from the encoded data
    output_token_idxs = (encoded_data['input_ids'] == output_token).nonzero()

    # we set the attention mask to 0 for all text after the "OUTPUT", since its what we want to predict.
    # This is how we tell the model to avoid that section when prediction, so it predicts only based on the test
    # that precedes the "OUTPUT" token.
    for idx, attn_mask in enumerate(encoded_data['attention_mask']):
        attn_mask[output_token_idxs[idx][1]:] = 0

    # we set everything that precedes the "OUTPUT:..." to -100. This is how we tell the model
    # to ignore that part when calculating the loss, so the loss is only w.r.t the generated text ("OUTPUT" onwards)
    tmp_labels = []
    for idx, input_id in enumerate(encoded_data['input_ids']):
        label = input_id.detach().clone()
        label[:output_token_idxs[idx][1]] = -100
        tmp_labels.append(label)

    batch = {'input_ids': torch.stack([result for result in encoded_data['input_ids']]),
             'attention_mask': torch.stack([result for result in encoded_data['attention_mask']]),
             'labels': torch.stack([result for result in tmp_labels])}
    return batch


class _ApartmentDataset(Dataset):
    def __init__(self, data_arr: List[str]):
        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/mGPT")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data: List[str] = data_arr
        self.tokenized_data: Dict[str, torch.Tensor] = _tokenize_data_and_labels(self.tokenizer, self.data)

    def __getitem__(self, index):
        """
        :return: both the tokenized data and labels and the original text
        """
        txt_val = self.data[index]
        dict_val = {
            'input_ids': self.tokenized_data['input_ids'][index],
            'attention_mask': self.tokenized_data['attention_mask'][index],
            'labels': self.tokenized_data['labels'][index]
        }
        return txt_val, dict_val

    def __len__(self):
        return len(self.data)


def get_dataset() -> Tuple[List[str], List[str], _ApartmentDataset, _ApartmentDataset, GPT2Tokenizer]:
    data = _load_data_from_jsonl()
    train_arr, test_arr = _train_test_split(data)
    train_ds = _ApartmentDataset(train_arr)
    test_ds = _ApartmentDataset(test_arr)
    return train_arr, test_arr, train_ds, test_ds, train_ds.tokenizer
