import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch
from rudalle import get_realesrgan, get_ruclip, get_rudalle_model, get_tokenizer, get_vae
from rudalle.pipelines import cherry_pick_by_clip, generate_images, super_resolution
from rudalle.utils import seed_everything

from utils import log, ArgumentParser

parser = ArgumentParser()
parser.add_argument("-chkpt")
parser.add_argument("-model", default="Malevich", choices={"Malevich", "Emojich"})
parser.add_argument("-texts", nargs="+")
parser.add_argument("-interpolate", default=0)
args = parser.parse_args()

with log("loading upsample GAN"):
    realesrgan = get_realesrgan("x2", device=args.device)
with log("loading CLiP"):
    ruclip, ruclip_processor = get_ruclip("ruclip-vit-base-patch32-v5")
    ruclip = ruclip.to(args.device)
tokenizer = get_tokenizer()


def sample(model, vae, text: str, target_folder: str, clean_prev: bool = False):
    seed_everything(42)
    pil_images, codebooks = [], []
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
        _pil_images, _, _codebooks = generate_images(
            text, tokenizer, model, vae, top_k=top_k, images_num=images_num, top_p=top_p
        )
        pil_images += _pil_images
        codebooks += _codebooks

    top_images, _ = cherry_pick_by_clip(pil_images, text, ruclip, ruclip_processor, device=args.device, count=24)
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
        model.load_state_dict(torch.load(f"./checkpoints/{args.chkpt}__{args.model}_last.pt"))
    with log("loading vae"):
        vae = get_vae().to(args.device)
else:
    with log("loading ruDALL-e model"):
        model = get_rudalle_model(args.model, pretrained=True, fp16=True, device=args.device)
    with log("loading vae"):
        vae = get_vae(dwt=True).to(args.device)

chkpt_name = args.chkpt if args.chkpt else "initial"

if args.texts == ["examples"]:
    args.texts = [
        "акварель",  # watercolor
        "аниме",  # anime
        "Бексиньский Здзислав картина",  # Beksinski Zdzislav picture
        "векторная графика",  # vector graphic
        "вышивка",  # embroidery
        "Гориллы",  # Gorillas
        "графический роман",  # graphic novel
        "детальная картина маслом",  # detailed oil painting
        "картина из ван гога",  # van gogh painting
        "киберпанк",  # cyberpunk
        "коллаж",  # collage
        "коммунистический социалистический молодежный боец",  # communist socialist youth fighter
        "линейное искусство",  # line art
        "лофи",  # lo-fi
        "милая сумасшедшая обезьяна",  # cute crazy monkey
        "Моне Клод картина",  # Monet Claude painting
        "оранжевый панк психоделическая картина",  # orange punk psychedelic painting
        "пиксельное искусство",  # pixel art
        "плоские цвета",  # flat colors
        "рисунок тушью",  # ink drawing
        "Сбой",  # Failure
        "Советский агитационный плакат",  # Soviet propaganda poster
        "средневековый пергамент",  # medieval parchment
        "трехмерный реалистичный рендер",  # three-dimensional realistic rendering
        "фотореалистичный",  # photorealistic
        "фотошоп",  # photoshop
        "хиппи бесплатно любовь солнышко",  # hippie free love sunshine
        "электрические цвета",  # electric colors
        "эффект линзы рыбий глаз",  # fisheye lens effect
        "мохнатая кожа",  # furry skin
        "лицо как у инопланетянина, реалистичный рендеринг",  # alien like face, realistic render
        "киноафиша",  # movie poster
        "психоделическая музыкальная обложка",  # psychedelic music cover
        "Графика в стиле модерн",  # Art Nouveau graphics
    ]

for text in args.texts:
    with log(f"Sample with {text}"):
        sample(model, vae, text, f"samples/{chkpt_name}__{args.model}/{text}", not args.cont)

if args.interpolate > 0:
    pass
