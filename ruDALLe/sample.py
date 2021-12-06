from argparse import ArgumentParser
import torch
from rich import print
from rudalle import get_rudalle_model, get_vae
from pathlib import Path
from rudalle import get_realesrgan, get_ruclip, get_tokenizer
from rudalle.pipelines import cherry_pick_by_clip, generate_images, super_resolution
from rudalle.utils import seed_everything
import shutil

parser = ArgumentParser()
parser.add_argument("-chkpt")
parser.add_argument("-texts", nargs="+")
parser.add_argument("-clean", action="store_true")
parser.add_argument("-device", default="cuda:1")
args = parser.parse_args()

print("[yellow]Loading upsample GAN[/yellow]")
realesrgan = get_realesrgan("x2", device=args.device)
print("[green]Loaded upsample GAN[/green]")
print("[yellow]Loading CLiP[/yellow]")
ruclip, ruclip_processor = get_ruclip("ruclip-vit-base-patch32-v5")
ruclip = ruclip.to(args.device)
print("[green]Loaded CLiP[/green]")
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

    top_images, clip_scores = cherry_pick_by_clip(pil_images, text, ruclip, ruclip_processor, device=args.device, count=24)
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
    model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=args.device)
    model.load_state_dict(torch.load(args.chkpt))
    vae = get_vae().to(args.device)
else:
    model = get_rudalle_model("Malevich", pretrained=True, fp16=True, device=args.device)
    vae = get_vae(dwt=True).to(args.device)

for text in args.texts:
    print(f"[red]{text}[/red]")
    sample(model, vae, text, f"generate/{text}", args.clean)
