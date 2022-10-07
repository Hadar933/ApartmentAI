import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from Dataset import get_dataset

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


def model_train(train_dataset, model, epochs=5, lr=2e-5, freeze=True):
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
            progressbar.set_description(f"Train Loss: {np.mean(train_loss[-10:]):.3f}")
    return model


def model_test(test_dataset, model):
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

        # print running average loss:
        progressbar.set_description(f"Train Loss: {np.mean(test_loss[-10:]):.3f}")
    return model


def _freeze_weights(model):
    """
    Freezes transformer layers except the first and the last one. Does not freeze any layer-norms
    :param model: a transformer gpt2 model
    """
    for n, p in model.named_parameters():
        if 'transformer.h' in n:
            layer_num = int(n.split('.')[2])
            if 'ln_' not in n and 0 < layer_num < 23:
                p.requires_grad = False


if __name__ == '__main__':
    train_data, test_data = get_dataset()
    gpt2 = GPT2LMHeadModel.from_pretrained('sberbank-ai/mGPT')
    # import torch.nn as nn
    #
    # conv = nn.Sequential(
    #     nn.Conv2d(1, 20, 5),
    #     nn.ReLU()
    # )
    model_train(train_data, gpt2)
