from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AdamW, GPT2Tokenizer
from Dataset import get_dataset, _ApartmentDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Accumulated batch size (since GPT2 is so big)
# def pack_tensor(new_dict: torch.Tensor, packed_dict: torch.Tensor, max_seq_len: int):
#     """
#
#     :param new_dict:
#     :param packed_dict:
#     :param max_seq_len:
#     :return:
#     """
#     if packed_dict is None:
#         return new_dict, True, 0
#     if new_dict['input_ids'].size()[1] + packed_dict['input_ids'].size()[1] > max_seq_len:
#         return packed_dict, False, new_dict
#     else:
#         packed_dict['input_ids'] = torch.cat([new_dict['input_ids'], packed_dict['input_ids'][:, 1:]], dim=1)
#         packed_dict['attention_mask'] = torch.cat([new_dict['attention_mask'], packed_dict['attention_mask'][:, 1:]],
#                                                   dim=1)
#         packed_dict['labels'] = torch.cat([new_dict['labels'], packed_dict['labels'][:, 1:]], dim=1)
#
#         return packed_dict, True, 0

def _freeze_weights(model: GPT2LMHeadModel):
    """
    Freezes transformer layers except the first and the last one. Does not freeze any layer-norms
    :param model: a transformer gpt2 model
    """
    for n, p in model.named_parameters():
        if 'transformer.h' in n:
            layer_num = int(n.split('.')[2])
            if 'ln_' not in n and 0 < layer_num < 23:
                p.requires_grad = False


def model_train(train_dataset: _ApartmentDataset, model: GPT2LMHeadModel, epochs=5, lr=2e-5, freeze=True) -> Tuple[
    GPT2LMHeadModel, List[str]]:
    """
    freezes most of the models weights and trains the model
    :return: the trained model and a list of loss values
    """
    if freeze: _freeze_weights(model)

    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    train_loss_arr = []

    for epoch in range(epochs):
        progressbar = tqdm(enumerate(train_dataloader))
        print(f"Training epoch {epoch}")

        for idx, entry_dict in progressbar:
            input_ids = entry_dict['input_ids'].to(device)
            attn_mask = entry_dict['attention_mask'].to(device)
            labels = entry_dict['labels'].to(device)

            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)

            train_loss = outputs[0]
            train_loss.backward()
            optimizer.step()
            train_loss_arr.append(train_loss.detach().item())

            optimizer.zero_grad()
            model.zero_grad()

            # print running average loss:
            progressbar.set_description(f"Train Loss: {train_loss:.3f}")
    return model, train_loss_arr


def model_test(test_dataset: _ApartmentDataset, model: GPT2LMHeadModel) -> List[float]:
    """
    tests the model after the training step
    :return: the test loss array
    """
    model = model.to(device)
    model.eval()

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    test_loss_arr = []

    progressbar = tqdm(enumerate(test_dataloader))

    for idx, entry_dict in progressbar:
        input_ids = entry_dict['input_ids'].to(device)
        attn_mask = entry_dict['attention_mask'].to(device)
        labels = entry_dict['labels'].to(device)

        outputs = model(input_ids, attention_mask=attn_mask, labels=labels)

        test_loss = outputs[0]
        test_loss_arr.append(test_loss.detach().item())

        progressbar.set_description(f"Test Loss: {test_loss:.3f}")
    return test_loss_arr


def generate_output(trained_model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer, data: List[str]):
    """

    :param trained_model:
    :param tokenizer:
    :param data:
    :return:
    """
    prompts = [f"INPUT: {item}. OUTPUT: " for item in data]
    results = []
    for text in prompts:
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        out = trained_model.generate(
            input_ids,
            min_length=100,
            max_length=100,
            eos_token_id=5,
            pad_token=1,
            top_k=10,
            top_p=0.0,
            no_repeat_ngram_size=5
        )
        generated_text = list(map(tokenizer.decode, out))[0]
        results.append(generated_text)
    for generated_text in results:
        print('---')
        print(generated_text)


if __name__ == '__main__':
    tr_text, te_text, train_data, test_data, tknzr = get_dataset()
    gpt2 = GPT2LMHeadModel.from_pretrained('sberbank-ai/mGPT')
    model_train(train_data, gpt2)
    model_test(test_data, gpt2)
    generate_output(gpt2, tknzr, te_text)
