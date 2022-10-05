import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from Dataset import get_dataset


# Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_dict: torch.Tensor, packed_dict: torch.Tensor, max_seq_len: int):
    """

    :param new_dict:
    :param packed_dict:
    :param max_seq_len:
    :return:
    """
    if packed_dict is None:
        return new_dict, True, 0
    if new_dict['input_ids'].size()[1] + packed_dict['input_ids'].size()[1] > max_seq_len:
        return packed_dict, False, new_dict
    else:
        packed_dict['input_ids'] = torch.cat([new_dict['input_ids'], packed_dict['input_ids'][:, 1:]], dim=1)
        packed_dict['attention_mask'] = torch.cat([new_dict['attention_mask'], packed_dict['attention_mask'][:, 1:]],
                                                  dim=1)
        packed_dict['labels'] = torch.cat([new_dict['labels'], packed_dict['labels'][:, 1:]], dim=1)

        return packed_dict, True, 0


def train(dataset, model, batch_size=16, epochs=5, lr=2e-5, warmup_steps=20, output_dir=".", output_prefix="wreckgar",
          save_model_on_epoch=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = 0
    accumulating_batch_count = 0
    input_dict = None
    # input_tensor = torch.rand((1, 12))
    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(f"loss: {loss}.")
        for idx, entry_dict in tqdm(enumerate(train_dataloader)):
            input_dict, carry_on, remainder = pack_tensor(entry_dict, input_dict, 768)

            if carry_on and idx != len(train_dataloader) - 1:
                continue

            input_ids = input_dict['input_ids'].to(device)
            attn_mask = input_dict['attention_mask'].to(device)
            labels = input_dict['labels'].to(device)
            # outputs = model(input_tensor, labels=input_tensor)
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            # accumulating_batch_count += 1
            input_dict = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


if __name__ == '__main__':
    train_data, test_data = get_dataset()
    gpt2 = GPT2LMHeadModel.from_pretrained('sberbank-ai/mGPT')
    # import torch.nn as nn
    #
    # conv = nn.Sequential(
    #     nn.Conv2d(1, 20, 5),
    #     nn.ReLU()
    # )
    train(train_data, gpt2)
