import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from Dataset import get_dataset
import time

# tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/mGPT')

# Accumulated batch size (since GPT2 is so big)
# def pack_tensor(new_tensor, packed_tensor, max_seq_len):
#     if packed_tensor is None:
#         return new_tensor, True, None
#     if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
#         return packed_tensor, False, new_tensor
#     else:
#         packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
#         return packed_tensor, True, None


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
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, value in tqdm(enumerate(train_dataloader)):
            input_ids = value['input_ids'].to(device)
            attn_mask = value['attention_mask'].to(device)
            labels = value['labels'].to(device)
            # (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)
            #
            # if carry_on and idx != len(train_dataloader) - 1:
            #     continue

            # input_tensor = input_tensor.to(device)
            # outputs = model(input_tensor, labels=input_tensor)
            outputs = model(input_ids, attention_mask=attn_mask, labels=labels)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


if __name__ == '__main__':
    train_data, test_data = get_dataset()
    start = time.time()
    model = GPT2LMHeadModel.from_pretrained('sberbank-ai/mGPT')
    end = time.time()
    train(train_data, model)
