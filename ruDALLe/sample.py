import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from rudalle import get_realesrgan, get_ruclip, get_rudalle_model, get_tokenizer, get_vae
from rudalle.pipelines import cherry_pick_by_clip, generate_images, super_resolution
from rudalle.utils import seed_everything

from utils import log

parser = ArgumentParser()
parser.add_argument("-chkpt")
parser.add_argument("-model", default="Malevich")
parser.add_argument("-texts", nargs="+")
parser.add_argument("-c", action="store_true")
parser.add_argument("-device", default="cuda:1")
args = parser.parse_args()

with log("loading upsample GAN"):
    realesrgan = get_realesrgan("x2", device=args.device)
with log("loading CLiP"):
    ruclip, ruclip_processor = get_ruclip("ruclip-vit-base-patch32-v5")
    ruclip = ruclip.to(args.device)
tokenizer = get_tokenizer()


def sample(model, vae, text: str, target_folder: str, clean_prev: bool = False):
    seed_everything(42)
    pil_images = []
    scores = []
    for top_k, top_p, images_num in [
        (2048, 0.995, 6),
        (1536, 0.99, 6),
        (1024, 0.99, 6),
        (1024, 0.98, 6),
        (512, 0.97, 6),
        (384, 0.96, 6),
        (256, 0.95, 6),
        (128, 0.95, 6),
    ]:
        _pil_images, _scores = generate_images(
            text, tokenizer, model, vae, top_k=top_k, images_num=images_num, top_p=top_p
        )
        pil_images += _pil_images
        scores += _scores

    top_images, clip_scores = cherry_pick_by_clip(
        pil_images, text, ruclip, ruclip_processor, device=args.device, count=24
    )
    sr_images = super_resolution(top_images, realesrgan)

    target_folder = Path(f"./{target_folder}/")
    if target_folder.exists() and clean_prev:
        shutil.rmtree(target_folder)
    target_folder.mkdir(exist_ok=True, parents=True)
    for image in sr_images:
        i = 0
        while (target_folder / f"{i:05}.png").exists():
            i += 1
        image.save(target_folder / f"{i:05}.png")


if args.chkpt:
    with log("loading checkpoint"):
        model = get_rudalle_model(args.model, pretrained=True, fp16=True, device=args.device)
        model.load_state_dict(torch.load(f"./checkpoints/{args.chkpt}_dalle_last.pt"))
    with log("loading vae"):
        vae = get_vae().to(args.device)
else:
    with log("loading ruDALL-e model"):
        model = get_rudalle_model(args.model, pretrained=True, fp16=True, device=args.device)
    with log("loading vae"):
        vae = get_vae(dwt=True).to(args.device)

chkpt_name = args.chkpt if args.chkpt else "initial"
for text in args.texts:
    with log(f"Sample with {text}"):
        sample(model, vae, text, f"samples/{chkpt_name}/{text}", not args.c)
