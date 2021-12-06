import os
import random
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision.transforms as T
from rudalle import get_rudalle_model, get_tokenizer, get_vae
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW

from utils import log

parser = ArgumentParser()
parser.add_argument("-dataset", default="asterix1to8_panels")
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-epochs", default=10, type=int)
parser.add_argument("-save_path", default="checkpoints/", type=Path)
parser.add_argument("-bs", default=4, type=int)
parser.add_argument("-clip", default=0.24, type=float)
parser.add_argument("-lr", default=8e-5, type=float)
args = parser.parse_args()
args.save_path.mkdir(exist_ok=True)

with log("loading base model"):
    model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=args.device)
with log("loading vae"):
    vae = get_vae().to(args.device)
tokenizer = get_tokenizer()


class RuDalleDataset(Dataset):
    clip_filter_thr = 0.24

    def __init__(
        self, file_path, csv_path, tokenizer, resize_ratio=0.75, shuffle=True, load_first=None, caption_score_thr=0.6
    ):
        self.text_seq_length = model.get_param("text_seq_length")
        self.tokenizer = tokenizer
        self.target_image_size = 256
        self.image_size = 256
        self.samples = []

        self.image_transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.RandomResizedCrop(
                    self.image_size, scale=(1.0, 1.0), ratio=(1.0, 1.0)  # в train было scale=(0.75., 1.),
                ),
                T.ToTensor(),
            ]
        )

        df = pd.read_csv(csv_path)
        for caption, f_path in zip(df["caption"], df["name"]):
            if len(caption) > 10 and len(caption) < 100 and os.path.isfile(f"{file_path}/{f_path}"):
                self.samples.append([file_path, f_path, caption])
        if shuffle:
            np.random.shuffle(self.samples)
            print("Shuffled")

    def __len__(self):
        return len(self.samples)

    def load_image(self, file_path, img_name):
        image = PIL.Image.open(f"{file_path}/{img_name}")
        return image

    def __getitem__(self, item):
        item = item % len(self.samples)  # infinite loop, modulo dataset size
        file_path, img_name, text = self.samples[item]
        try:
            image = self.load_image(file_path, img_name)
            image = self.image_transform(image).to(args.device)
        except Exception as err:  # noqa
            print(err)
            random_item = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(random_item)
        text = tokenizer.encode_text(text, text_seq_length=self.text_seq_length).squeeze(0).to(args.device)
        return text, image


total_seq_length = model.get_param("total_seq_length")

st = RuDalleDataset(file_path=f"datasets/{args.dataset}/", csv_path=f"datasets/{args.dataset}.csv", tokenizer=tokenizer)
train_dataloader = DataLoader(st, batch_size=args.bs, shuffle=True, drop_last=True)


model.train()
optimizer = AdamW(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr, final_div_factor=500, steps_per_epoch=len(train_dataloader), epochs=args.epochs
)


def freeze(
    model,
    freeze_emb=True,
    freeze_ln=False,
    freeze_attn=False,
    freeze_ff=True,
    freeze_other=True,
):
    for name, p in model.module.named_parameters():
        name = name.lower()
        if "ln" in name or "norm" in name:
            p.requires_grad = not freeze_ln
        elif "embeddings" in name:
            p.requires_grad = not freeze_emb
        elif "mlp" in name:
            p.requires_grad = not freeze_ff
        elif "attn" in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
    return model


# markdown Simple training loop
def train(model, train_dataloader: RuDalleDataset):
    loss_logs = []
    progress = tqdm(total=len(train_dataloader), desc="finetuning goes brrr")
    save_counter = 0
    for epoch in range(args.epochs):
        for text, images in train_dataloader:
            device = model.get_param("device")
            save_counter += 1
            model.zero_grad()
            attention_mask = torch.tril(torch.ones((args.bs, 1, total_seq_length, total_seq_length), device=device))
            image_input_ids = vae.get_codebook_indices(images)

            input_ids = torch.cat((text, image_input_ids), dim=1)
            loss, loss_values = model.forward(input_ids, attention_mask, return_loss=True)
            # train step
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_logs += [loss.item()]
            progress.update()
            progress.set_postfix({"loss": loss.item()})

    print(f"Complitly tuned and saved here  {args.dataset}__dalle_last.pt")

    plt.plot(loss_logs)
    plt.savefig(args.save_path / f"{args.dataset}_dalle_last.png")

    torch.save(model.state_dict(), args.save_path / f"{args.dataset}_dalle_last.pt")


# @markdown You can unfreeze or freeze more parametrs, but it can
model = freeze(
    model=model, freeze_emb=False, freeze_ln=False, freeze_attn=True, freeze_ff=True, freeze_other=False
)  # freeze params to

train(model, train_dataloader)
