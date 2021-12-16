# TODO
# - match times to beats
# - accept beat counts instead of times
# - make audio feature outputter

import argparse
import math
import os
import queue
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import clip
import imageio
import librosa
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
VERBOSE = False


def parse_cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    relevant = parser.add_argument_group("Basic arguments")
    relevant.add_argument("init_image", type=str, help="Path to the inital frame")
    relevant.add_argument("-script", type=str, default=None, help="Path to the CSV prompt script")
    relevant.add_argument("-music", type=str, default=None, help="Path to the music file")
    relevant.add_argument("-lr", type=float, default=0.1, help="Learning rate")
    relevant.add_argument(
        "-lr_warmup", type=float, default=0.05, help="Percentage of the video that is used for linear LR start up"
    )
    relevant.add_argument("-device", type=str, default="cuda:0", help="GPU to use")
    relevant.add_argument("-fps", type=int, default=30, help="Sampling rate alas the FPS of the output video")

    effects = parser.add_argument_group("Effect arguments")
    effects.add_argument("-max_zoom", type=float, default=0.1, help="Maximal zoom percentage in one step")
    effects.add_argument("-max_rotate", type=int, default=0, help="Maximal degrees rotation in one step")
    # effects.add_argument("-beat_measure", type=int, default=4, help="Measure of the song, counted in beats")
    # effects.add_argument("-beat_phase", type=int, default=32, help="Start of the first measure, counted in beats")
    effects.add_argument("-pulse_delay", type=float, default=0.5, help="Delay of the memory pulse in beats")

    irrelevant = parser.add_argument_group("Probably uninteresting arguments")
    irrelevant.add_argument(
        "-size", type=int, nargs=2, default=None, help="Output width/height of the video, otherwise base on init frame"
    )
    irrelevant.add_argument("-seed", type=int, default=None, help="torch seed to use")
    irrelevant.add_argument("-optimizer", type=str, default="Adam", help="Optimizer to use from torch.optim")
    irrelevant.add_argument(
        "-save_path", type=str, default="./steps", help="Path to the folder where frames are saved"
    )
    irrelevant.add_argument(
        "-keep_old_frames", action="store_true", help="Do not clear out the save frame folder in the beginning"
    )
    irrelevant.add_argument(
        "-extra_steps_after_image_effect", type=int, default=4, help="extra steps after image effect"
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
    irrelevant.add_argument("-eq_bins", type=int, default=16, help="Number of bins for the spectrogram")
    irrelevant.add_argument("-v", action="store_true", help="More verbosity", dest="verbose")

    return parser.parse_args()


@contextmanager
def log(action: str, suppress_output: bool = not VERBOSE):
    with open(os.devnull, "w") as devnull:
        start_time = time.perf_counter()
        print(f"[yellow]{action}…[/yellow]")
        if suppress_output:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
        try:
            yield None
        finally:
            if suppress_output:
                sys.stdout, sys.stderr = old_stdout, old_stderr
            duration = time.perf_counter() - start_time
            print(f"[green]{action} ✔[/green] ({int(duration)}s)")


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
    img = Image.open(path).convert("RGB")
    size = img.size if cli.size is None else cli.size
    img = img.resize(((size[0] // f) * f, (size[1] // f) * f), Image.LANCZOS)
    return TF.to_tensor(img)


def image_effects(model, z, img, zoom, rotate):
    def zoom_at(img, x, y, zoom):
        w, h = img.size
        zoom2 = zoom * 2
        img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))
        return img.resize((w, h), Image.LANCZOS)

    pil_image = Image.fromarray(np.array(img).astype("uint8"), "RGB")
    if cli.max_zoom > 0:
        pil_image = zoom_at(pil_image, img.shape[1] / 2, img.shape[0] / 2, 1 + zoom * cli.max_zoom)
    if cli.max_rotate > 0:
        pil_image = pil_image.rotate(round(rotate * cli.max_rotate))

    _z, *_ = model.encode(TF.to_tensor(pil_image).to(cli.device).unsqueeze(0) * 2 - 1)
    z.data = _z.data
    return z


def load_script(input_name):
    script = {}
    if input_name is not None:
        script = pd.read_csv(f"../inputs/{input_name}", header=None, index_col=0)
        beat_indexed = True
        if script.index.dtype != int:
            beat_indexed = False
            script.index = cli.fps * script.index.to_series().apply(
                lambda time: sum(x * int(t) for x, t in zip([60, 1], time.split(":")))
            )
        script: Dict[int, str] = script.to_dict()[1]
    return script, beat_indexed


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
    return eq


def compute_beat_markers(y, sr, hop_length=512):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length).T

    tempo, btrack = librosa.beat.beat_track(onset_envelope=onset_env)
    times = librosa.times_like(onset_env, sr=sr)

    beats = np.zeros(round(y.size / sr * cli.fps) + 1)
    beats[np.round(times[btrack] * cli.fps).astype(int)] = 1
    beats_mag = beats.copy()
    beats_mag[np.round(times[btrack] * cli.fps).astype(int)] = onset_env[btrack]
    beats_mag = np.convolve(beats_mag, [0, 0, 0, 0, 0, 0, 0, 1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01], mode="same")
    return tempo, beats, beats_mag


def load_music_features(input_name: str, eq_bins):
    wave, sr = librosa.load(f"../inputs/{input_name}")
    eq = compute_eq(wave, sr, amax=35, eq_bins=eq_bins)
    tempo, beats, beats_mag = compute_beat_markers(wave, sr)
    return tempo, beats, beats_mag, eq


def train_step(save: bool = True, lr_step: bool = True) -> np.ndarray:
    optimizer.zero_grad(set_to_none=True)
    z_q = vector_quantize(model, z)
    y = clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)
    y_aug = perceptor.encode_image(normalize(make_cutouts(y))).float()
    loss = prompt(y_aug)
    loss.backward()
    optimizer.step()
    with torch.inference_mode():
        z.copy_(z.maximum(z_min).minimum(z_max))
    if lr_step:
        scheduler.step()
    img: np.ndarray = y.cpu().detach().mul(255).clamp(0, 255)[0].numpy().astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    if save:
        imageio.imwrite(save_path / f"{frame:05d}.jpg", img)
    return img


if __name__ == "__main__":
    cli = parse_cli()
    assert (cli.script is not None) or (cli.music is not None)

    seed = torch.seed() if cli.seed is None else cli.seed
    torch.manual_seed(seed)
    save_path = Path(cli.save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    if not cli.keep_old_frames:
        print("[red]Deleting old frames…[/red]")
        for p in filter(Path.is_file, save_path.iterdir()):
            p.unlink()

    script, beat_indexed = load_script(cli.script)
    if cli.music is not None:
        tempo, beats, beats_mag, eq = load_music_features(cli.music, cli.eq_bins)

    with log("Load image model"):
        model, z_min, z_max = load_vqgan_model(cli.vqgan_config, cli.vqgan_checkpoint)
    with log("Load clip"):
        perceptor = clip.load(cli.clip_model, jit=False)[0].eval().requires_grad_(False).to(cli.device)

    n_frames = max(script.keys()) if cli.music is None else eq.shape[-1]
    make_cutouts = MakeCutouts(perceptor.visual.input_resolution, cli.cutn, cut_pow=cli.cut_pow)
    z, *_ = model.encode(load_init_image(model, f"../inputs/{cli.init_image}").to(cli.device).unsqueeze(0) * 2 - 1)
    z.requires_grad_(True)
    optimizer: torch.optim.Optimizer = getattr(torch.optim, cli.optimizer)([z], lr=cli.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda x: np.minimum(1, x / (n_frames * cli.lr_warmup))
    )
    delay_queue = queue.Queue(round(cli.pulse_delay * cli.fps * 60 / tempo))
    z_anchor = None
    c_z = z.shape[1]  # number of latent dimensions!

    console.rule("🕹 Settings 🕹")
    print(
        f"""[yellow]
        The plan is the following:
        Will generate {n_frames=} on {cli.device=} with {seed=}.
        Learning rate with reach {cli.lr=} after linearly going there for {n_frames*cli.lr_warmup} steps.
        Prompt script is[/yellow]"""
    )
    print(script)
    print(
        f"""[yellow]
        With the {cli.music=} that has {sum(beats==1)} zoomable beats at {tempo=}.[/yellow]"""
    )
    console.rule("🐢🦖 Will start now ☘️🍀")

    beat_idx = 0
    for frame in trange(n_frames):
        if not beat_indexed and frame in script:
            prompt = make_prompt(script[frame])

        img = train_step()

        if delay_queue.full():
            z_anchor = delay_queue.get()
        delay_queue.put(z.detach().clone())

        if cli.music is not None:
            if z_anchor is not None and eq[0, frame] > 0.3:
                with torch.inference_mode():
                    z_translation = z_anchor - z
                    z[:, :, :, :] += 0.1 * eq[0, frame] * z_translation[:, :, :, :]

            if beats[frame] == 1 and (cli.max_zoom > 0 or cli.max_rotate > 0):
                beat_idx += 1
                if beat_indexed and beat_idx in script:
                    print(f"Change to {script[frame]}")
                    prompt = make_prompt(script[frame])
                z = image_effects(model, z, img, eq[0, frame] * beats_mag[frame], eq[-4, frame] * beats_mag[frame])
                for _ in range(cli.extra_steps_after_image_effect):
                    train_step(save=False, lr_step=False)
                