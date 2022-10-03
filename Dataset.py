import json
from typing import List, Dict, Tuple


def load_labels_from_jsonl(jsonl_path='all.jsonl') -> List[Tuple[Dict, str]]:
    """
    extracts all the relevant labels from the jsonl file
    :param jsonl_path: str path to the jsonl labels file
    :return: a list in which every element is a tuple of dictionary of labels and a string that concatenates
             those labels to a single string
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
        all_labels.append((label_dict, string_output))
    return all_labels


if __name__ == '__main__':
    labels = load_labels_from_jsonl()
    for l in labels:
        print(l[1])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
