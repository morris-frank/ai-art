from argparse import ArgumentParser
import torch
from . import console


def do_diffusion(inputs, device):
    from ._clip import CLIP
    from .diffusion import diffusion_sample

    console.log("Load CLIP")
    clip = CLIP(device)
    console.log("Loaded CLIP")
    target = clip.prompts2embeddings(inputs)
    console.log(f"Embedded targets, {target.shape}")
    console.log(inputs)
    diffusion_sample(target=target, clip=clip, device=device)


def main(inputs: list[str], device: str, model: str, seed: int = None):
    console.log("Entered main")
    if seed is not None:
        torch.manual_seed(seed)

    if model == "diffusion":
        do_diffusion(inputs, device)


parser = ArgumentParser()
parser.add_argument("inputs", nargs="+", help="Positive inputs to use (path to image, path to script or prompt)")
parser.add_argument("-seed", default=0)
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-model", default="diffusion")

main(**vars(parser.parse_args()))
