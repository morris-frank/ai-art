import argparse
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict


def parse_cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    relevant = parser.add_argument_group("Basic arguments")
    relevant.add_argument("init_image", type=str, help="Path to the inital frame")
    relevant.add_argument("-script", type=str, default=None, help="Path to the CSV prompt script")
    relevant.add_argument("-music", type=str, default=None, help="Path to the music file")
    relevant.add_argument("-lr", type=float, default=0.1, help="Learning rate")
    relevant.add_argument("-device", type=str, default="cuda:0", help="GPU to use")
    relevant.add_argument("-fps", type=int, default=25, help="Sampling rate alas the FPS of the output video")

    effects = parser.add_argument_group("Effect arguments")
    effects.add_argument("-max_zoom", type=float, default=0.01, help="Maximal zoom percentage in one step")
    effects.add_argument("-max_rotate", type=int, default=5, help="Maximal degrees rotation in one step")
    effects.add_argument("-beat_measure", type=int, default=16, help="Measure of the song, counted in beats")
    effects.add_argument("-beat_phase", type=int, default=5, help="Start of the first measure, counted in beats")

    irrelevant = parser.add_argument_group("Probably uninteresting arguments")
    irrelevant.add_argument("-size", type=int, nargs=2, default=[1280, 720], help="Output width/height of the video")
    irrelevant.add_argument("-seed", type=int, default=None, help="torch seed to use")
    irrelevant.add_argument("-optimizer", type=str, default="Adam", help="Optimizer to use from torch.optim")
    irrelevant.add_argument(
        "-save_path", type=str, default="./frames", help="Path to the folder where frames are saved"
    )
    irrelevant.add_argument(
        "-keep_old_frames", action="store_true", help="Do not clear out the save frame folder in the beginning"
    )
    irrelevant.add_argument("-cutn", type=int, default=32, help="Number of augmented cutouts made for ascending")
    irrelevant.add_argument("-cut_pow", type=float, default=1.0, help="cutout power")
    irrelevant.add_argument(
        "-vqgan_config",
        type=str,
        default="weights/vqgan_imagenet_f16_16384.yaml",
        help="Path to the config of the VQ-GAN",
    )
    irrelevant.add_argument(
        "-vqgan_checkpoint",
        type=str,
        default="weights/vqgan_imagenet_f16_16384.ckpt",
        help="Path to the weights of the VQ-GAN",
    )
    irrelevant.add_argument("-clip_model", type=str, default="ViT-B/32", help="CliP model to use")

    return parser.parse_args()


cli = parse_cli()
assert (cli.script is not None) or (cli.music is not None)


import clip
import imageio
import numpy as np
import pandas as pd
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from rich import print
from rich.console import Console
from taming.models import cond_transformer, vqgan
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from tqdm import trange

console = Console()


@contextmanager
def log(action: str):
    with open(os.devnull, "w") as devnull:
        start_time = time.perf_counter()
        print(f"[yellow]{action}…[/yellow]")
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield None
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            duration = time.perf_counter() - start_time
            print(f"[green]{action} ✔[/green] ({int(duration)}s)")


with log("Setup inputs"):
    seed = torch.seed() if cli.seed is None else cli.seed
    torch.manual_seed(seed)
    save_path = Path(cli.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    if not cli.keep_old_frames:
        print("[red]Deleting old frames…[/red]")
        for p in filter(Path.is_file, save_path.iterdir()):
            p.unlink()

    script = {}
    if cli.script is not None:
        script = pd.read_csv(cli.script, header=None, index_col=0)
        script.index = cli.fps * script.index.to_series().apply(
            lambda time: sum(x * int(t) for x, t in zip([60, 1], time.split(":")))
        )
        script: Dict[int, str] = script.to_dict()[1]

    if cli.music is not None:
        import librosa

        wave, sr = librosa.load(cli.music)

        def compute_eq(wave, sr, amax, eq_bins):
            C = np.abs(librosa.cqt(wave, sr=sr, fmin=librosa.note_to_hz("A1")))
            freqs = librosa.cqt_frequencies(C.shape[0], fmin=librosa.note_to_hz("A1"))
            perceptual_CQT = librosa.perceptual_weighting(C ** 2, freqs, ref=np.max)
            freqs = perceptual_CQT[2:-2, :]
            eq = None
            for i in range(eq_bins):
                l = freqs[5 * i : 5 * (i + 1), :].mean(0)
                l = librosa.resample(l, sr * l.shape[0] / wave.shape[0], cli.fps)
                l = l.clip(-80, -amax) + amax
                l -= l.min()
                l /= l.max()
                if eq is None:
                    eq = np.zeros((eq_bins, l.shape[-1]))
                eq[i, :] = l / l.max()
            eq -= np.median(eq, 1, keepdims=True)
            return eq

        def compute_beat_markers(wave, sr, osize):
            _, beat_idx = librosa.beat.beat_track(y=wave, sr=sr)
            beat_idx = np.round(librosa.frames_to_time(beat_idx, sr=sr) * cli.fps).astype(int)
            beats = np.zeros(osize)
            beats[beat_idx[cli.beat_phase :: cli.beat_measure]] = 1
            # beats = np.convolve(beats, [0, 0, 0, 0, 0, 0, 0, 1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01], mode="same")
            return beats

        eq = compute_eq(wave, sr, amax=35, eq_bins=16)
        beats = compute_beat_markers(wave, sr, eq.shape[1])
        del wave, sr, compute_eq, compute_beat_markers

with log("General library"):
    normalize = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )

    def resize_image(image, out_size):
        ratio = image.size[0] / image.size[1]
        area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
        size = round((area * ratio) ** 0.5), round((area / ratio) ** 0.5)
        return image.resize(size, Image.LANCZOS)

    class ClampWithGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, min, max):
            ctx.min = min
            ctx.max = max
            ctx.save_for_backward(input)
            return input.clamp(min, max)

        @staticmethod
        def backward(ctx, grad_in):
            (input,) = ctx.saved_tensors
            return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

    clamp_with_grad = ClampWithGrad.apply

    class ReplaceGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_forward, x_backward):
            ctx.shape = x_backward.shape
            return x_forward

        @staticmethod
        def backward(ctx, grad_in):
            return None, grad_in.sum_to_size(ctx.shape)

    replace_grad = ReplaceGrad.apply

    def resample(input, size, align_corners=True):
        def lanczos(x, a):
            def sinc(x):
                return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))

            cond = torch.logical_and(-a < x, x < a)
            out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
            return out / out.sum()

        def ramp(ratio, width):
            n = math.ceil(width / ratio + 1)
            out = torch.empty([n])
            cur = 0
            for i in range(out.shape[0]):
                out[i] = cur
                cur += ratio
            return torch.cat([-out[1:].flip([0]), out])[1:-1]

        n, c, h, w = input.shape
        dh, dw = size

        input = input.view([n * c, 1, h, w])

        if dh < h:
            kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
            pad_h = (kernel_h.shape[0] - 1) // 2
            input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
            input = F.conv2d(input, kernel_h[None, None, :, None])

        if dw < w:
            kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
            pad_w = (kernel_w.shape[0] - 1) // 2
            input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
            input = F.conv2d(input, kernel_w[None, None, None, :])

        input = input.view([n, c, h, w])
        return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)

    class MakeCutouts(nn.Module):
        def __init__(self, cut_size, cutn, cut_pow=1.0):
            super().__init__()
            self.cut_size = cut_size
            self.cutn = cutn
            self.cut_pow = cut_pow

        def forward(self, input):
            sideY, sideX = input.shape[2:4]
            max_size = min(sideX, sideY)
            min_size = min(sideX, sideY, self.cut_size)
            cutouts = []
            for _ in range(self.cutn):
                size = int(torch.rand([]) ** self.cut_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            return clamp_with_grad(torch.cat(cutouts, dim=0), 0, 1)

    class Prompt(nn.Module):
        def __init__(self, embed, weight=1.0, stop=float("-inf")):
            super().__init__()
            self.register_buffer("embed", embed)
            self.register_buffer("weight", torch.as_tensor(weight))
            self.register_buffer("stop", torch.as_tensor(stop))

        def forward(self, input):
            input_normed = F.normalize(input.unsqueeze(1), dim=2)
            embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
            dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
            dists = dists * self.weight.sign()
            return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


with log("Special library"):

    def load_vqgan_model(config_path, checkpoint_path):
        config = OmegaConf.load(config_path)
        if config.model.target == "taming.models.vqgan.VQModel":
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == "taming.models.vqgan.GumbelVQ":
            model = vqgan.GumbelVQ(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == "taming.models.cond_transformer.Net2NetTransformer":
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f"unknown model type: {config.model.target}")
        del model.loss
        model = model.to(cli.device)

        if config.model.target == "taming.models.vqgan.GumbelVQ":
            z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
        return model, z_min, z_max

    def vector_quantize(model, z):
        def _quantize(x, codebook):
            d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
            indices = d.argmin(-1)
            x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
            return replace_grad(x_q, x)

        if hasattr(model.quantize, "embed"):
            codebook = model.quantize.embed.weight
        else:
            codebook = model.quantize.embedding.weight
        z_q = _quantize(z.movedim(1, 3), codebook).movedim(3, 1)
        return z_q

    def make_prompt(txt):
        embed = perceptor.encode_text(clip.tokenize(txt).to(cli.device)).float()
        return Prompt(embed).to(cli.device)

    def load_init_image(model, path):
        f = 2 ** (model.decoder.num_resolutions - 1)
        img = Image.open(path).convert("RGB").resize(((cli.size[0] // f) * f, (cli.size[1] // f) * f), Image.LANCZOS)
        return TF.to_tensor(img)


with log("Effect library"):

    def image_effects(model, z, img, zoom, rotate):
        def zoom_at(img, x, y, zoom):
            w, h = img.size
            zoom2 = zoom * 2
            img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))
            return img.resize((w, h), Image.LANCZOS)

        if zoom == 0:
            return z

        pil_image = Image.fromarray(np.array(img).astype("uint8"), "RGB")
        pil_image = zoom_at(pil_image, img.shape[0] / 2, img.shape[1] / 2, 1 - zoom * cli.max_zoom)
        # pil_image = pil_image.rotate(round(rotate * cli.max_rotate))

        _z, *_ = model.encode(TF.to_tensor(pil_image).to(cli.device).unsqueeze(0) * 2 - 1)
        z.data = _z.data
        return z


with log("Load image model"):
    model, z_min, z_max = load_vqgan_model(cli.vqgan_config, cli.vqgan_checkpoint)
with log("Load clip"):
    perceptor = clip.load(cli.clip_model, jit=False)[0].eval().requires_grad_(False).to(cli.device)
with log("Setup latent, input and  optim"):
    make_cutouts = MakeCutouts(perceptor.visual.input_resolution, cli.cutn, cut_pow=cli.cut_pow)
    z, *_ = model.encode(load_init_image(model, cli.init_image).to(cli.device).unsqueeze(0) * 2 - 1)
    z.requires_grad_(True)
    optimizer: torch.optim.Optimizer = getattr(torch.optim, cli.optimizer)([z], lr=cli.lr)
    n_frames = max(script.keys()) if cli.music is None else eq.shape[-1]

console.rule("🕹 Settings 🕹")
print(f"""[yellow]
    The plan is the following:
    Will generate {n_frames=} on {cli.device=} with {seed=}\n
    using script:[/yellow]""")
print(script)
print(f"""[yellow]
    With the {cli.music=} that has {sum(beats)=} zoomable.[/yellow]""")
console.rule("🐢🦖 Will start now ☘️🍀")

# %% One big trainings loop
for frame in trange(n_frames):
    if frame in script:
        prompt = make_prompt(script[frame])

    optimizer.zero_grad(set_to_none=True)
    z_q = vector_quantize(model, z)
    y = clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)
    y_aug = perceptor.encode_image(normalize(make_cutouts(y))).float()
    loss = prompt(y_aug)
    loss.backward()
    optimizer.step()
    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))

    img: np.ndarray = y.cpu().detach().mul(255).clamp(0, 255)[0].numpy().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    imageio.imwrite(save_path / f"{frame:05d}.png", img)

    if cli.music is not None:
        z = image_effects(model, z, img, beats[frame], eq[-4, frame])
