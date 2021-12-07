import gc
import math
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial

import clip
import lpips
import torch
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from utils import file_input, log, output_path, print

print("Finished import")


def parse_prompt(prompt):
    if prompt.startswith("http://") or prompt.startswith("https://"):
        vals = prompt.rsplit(":", 2)
        vals = [vals[0] + ":" + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(":", 1)
    vals = vals + ["", "1"][len(vals) :]
    return vals[0], float(vals[1])


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
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff ** 2 + y_diff ** 2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock(
                [
                    nn.AvgPool2d(2),
                    ConvBlock(c, c * 2),
                    ConvBlock(c * 2, c * 2),
                    SkipBlock(
                        [
                            nn.AvgPool2d(2),
                            ConvBlock(c * 2, c * 4),
                            ConvBlock(c * 4, c * 4),
                            SkipBlock(
                                [
                                    nn.AvgPool2d(2),
                                    ConvBlock(c * 4, c * 8),
                                    ConvBlock(c * 8, c * 4),
                                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                ]
                            ),
                            ConvBlock(c * 8, c * 4),
                            ConvBlock(c * 4, c * 2),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        ]
                    ),
                    ConvBlock(c * 4, c * 2),
                    ConvBlock(c * 2, c),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ]
            ),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock(
                [
                    self.down,
                    ConvBlock(cs[0], cs[1]),
                    ConvBlock(cs[1], cs[1]),
                    SkipBlock(
                        [
                            self.down,
                            ConvBlock(cs[1], cs[2]),
                            ConvBlock(cs[2], cs[2]),
                            SkipBlock(
                                [
                                    self.down,
                                    ConvBlock(cs[2], cs[3]),
                                    ConvBlock(cs[3], cs[3]),
                                    SkipBlock(
                                        [
                                            self.down,
                                            ConvBlock(cs[3], cs[4]),
                                            ConvBlock(cs[4], cs[4]),
                                            SkipBlock(
                                                [
                                                    self.down,
                                                    ConvBlock(cs[4], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[4]),
                                                    self.up,
                                                ]
                                            ),
                                            ConvBlock(cs[4] * 2, cs[4]),
                                            ConvBlock(cs[4], cs[3]),
                                            self.up,
                                        ]
                                    ),
                                    ConvBlock(cs[3] * 2, cs[3]),
                                    ConvBlock(cs[3], cs[2]),
                                    self.up,
                                ]
                            ),
                            ConvBlock(cs[2] * 2, cs[2]),
                            ConvBlock(cs[2], cs[1]),
                            self.up,
                        ]
                    ),
                    ConvBlock(cs[1] * 2, cs[1]),
                    ConvBlock(cs[1], cs[0]),
                    self.up,
                ]
            ),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


parser = ArgumentParser()
parser.add_argument("-attention_resolutions", default="32, 16, 8")
parser.add_argument("-device", default="cuda:0")
parser.add_argument("-diffusion_steps", default=1000, type=int)
parser.add_argument("-image_size", default=512, type=int, choices=[512, 256])
parser.add_argument("-noise_schedule", default="linear")
parser.add_argument("-class_cond", action="store_true")
parser.add_argument("-not_learn_sigma", action="store_true")
parser.add_argument("-not_resblock_updown", action="store_true")
parser.add_argument("-not_use_fp16", action="store_true")
parser.add_argument("-not_use_scale_shift_norm", action="store_true")
parser.add_argument("-num_channels", default=256, type=int)
parser.add_argument("-num_head_channels", default=64, type=int)
parser.add_argument("-num_res_blocks", default=2, type=int)
parser.add_argument("-rescale_timesteps", action="store_true")
parser.add_argument("-timestep_respacing", default="500")
parser.add_argument("-use_checkpoint", action="store_true")
parser.add_argument("-use_version_one", action="store_true")
parser.add_argument("-tv_scale", default=0, type=float)  # Controls the smoothness of the final output.
parser.add_argument("-range_scale", default=0, type=int)  # Controls how far out of range RGB values are allowed to be.
parser.add_argument("-cutn", default=64, type=int)
parser.add_argument("-cut_pow", default=0.5, type=float)
parser.add_argument("-n_batches", default=1, type=int)
parser.add_argument("-seed", default=0, type=int)

parser.add_argument("-prompts", default=["a avocado logo"], nargs="*")
parser.add_argument("-image_prompts", default=[], nargs="*")
parser.add_argument(
    "-init_image", default=None, type=file_input
)  # This can be an URL or Colab local path and must be in quotes.
parser.add_argument(
    "-skip_timesteps", default=0, type=int
)  # This needs to be between approx. 200 and 500 when using an init image. Higher values make the output look more like the init.
parser.add_argument(
    "-init_scale", default=0, type=int
)  # This enhances the effect of the init image, a good value is 1000.
parser.add_argument(
    "-clip_guidance_scale", default=1500, type=int
)  # Controls how much the image should look like the prompt.
parser.add_argument("-batch_size", default=1, type=int)

args = parser.parse_args()
side_x = side_y = args.image_size, args.image_size

model_config = model_and_diffusion_defaults()
model_config.update(
    {
        "attention_resolutions": args.attention_resolutions,
        "class_cond": args.class_cond,
        "diffusion_steps": args.diffusion_steps,
        "image_size": args.image_size,
        "learn_sigma": not args.not_learn_sigma,
        "noise_schedule": args.noise_schedule,
        "num_channels": args.num_channels,
        "num_head_channels": args.num_head_channels,
        "num_res_blocks": args.num_res_blocks,
        "resblock_updown": not args.not_resblock_updown,
        "rescale_timesteps": args.rescale_timesteps,
        "timestep_respacing": args.timestep_respacing,
        "use_checkpoint": args.use_checkpoint,
        "use_fp16": not args.not_use_fp16,
        "use_scale_shift_norm": not args.not_use_scale_shift_norm,
    }
)

with log("loading diffusion UNet"):
    model, diffusion = create_model_and_diffusion(**model_config)
with log("load UNet weight"):
    if args.image_size == 512:
        model.load_state_dict(torch.load("weights/512x512_diffusion_uncond_finetune_008100.pt", map_location="cpu"))
    elif args.image_size == 256:
        model.load_state_dict(torch.load("weights/256x256_diffusion_uncond.pt", map_location="cpu"))
model.requires_grad_(False).eval().to(args.device)
if model_config["use_fp16"]:
    model.convert_to_fp16()

with log("loading secondary model"):
    if args.use_version_one:
        secondary_model = SecondaryDiffusionImageNet()
        secondary_model.load_state_dict(torch.load("weights/secondary_model_imagenet.pth", map_location="cpu"))
    else:
        secondary_model = SecondaryDiffusionImageNet2()
        secondary_model.load_state_dict(torch.load("weights/secondary_model_imagenet_2.pth", map_location="cpu"))
    secondary_model.eval().requires_grad_(False).to(args.device)

with log("loading CLiP"):
    clip_model = clip.load("ViT-B/16", jit=False)[0].eval().requires_grad_(False).to(args.device)
clip_size = clip_model.visual.input_resolution
normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
with log("loading LPIPS"):
    lpips_model = lpips.LPIPS(net="vgg").to(args.device)


def do_run():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    make_cutouts = MakeCutouts(clip_size, args.cutn, args.cut_pow)
    side_x = side_y = model_config["image_size"]

    target_embeds, weights = [], []

    for prompt in args.prompts:
        txt, weight = parse_prompt(prompt)
        target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(args.device)).float())
        weights.append(weight)

    for prompt in args.image_prompts:
        path, weight = parse_prompt(prompt)
        img = Image.open(file_input(path)).convert("RGB")
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(args.device))
        embed = clip_model.encode_image(normalize(batch)).float()
        target_embeds.append(embed)
        weights.extend([weight / args.cutn] * args.cutn)

    target_embeds = torch.cat(target_embeds)
    weights = torch.tensor(weights, device=args.device)
    if weights.sum().abs() < 1e-3:
        raise RuntimeError("The weights must not sum to 0.")
    weights /= weights.sum().abs()

    init = None
    if args.init_image is not None:
        init = Image.open(args.init_image).convert("RGB")
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(args.device).unsqueeze(0).mul(2).sub(1)

    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            n = x.shape[0]
            alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=args.device, dtype=torch.float32)
            sigma = torch.tensor(
                diffusion.sqrt_one_minus_alphas_cumprod[cur_t], device=args.device, dtype=torch.float32
            )
            cosine_t = alpha_sigma_to_t(alpha, sigma)
            pred = secondary_model(x, cosine_t[None].repeat([n])).pred
            clip_in = normalize(make_cutouts(pred.add(1).div(2)))
            image_embeds = clip_model.encode_image(clip_in).float()
            dists = spherical_dist_loss(image_embeds.unsqueeze(1), target_embeds.unsqueeze(0))
            dists = dists.view([args.cutn, n, -1])
            clip_losses = dists.mul(weights).sum(2).mean(0)
            tv_losses = tv_loss(pred)
            range_losses = range_loss(pred)
            loss = (
                clip_losses.sum() * args.clip_guidance_scale
                + tv_losses.sum() * args.tv_scale
                + range_losses.sum() * args.range_scale
            )
            if init is not None and args.init_scale:
                init_losses = lpips_model(pred, init)
                loss = loss + init_losses.sum() * args.init_scale
            grad = -torch.autograd.grad(loss, x)[0]
            return grad

    if model_config["timestep_respacing"].startswith("ddim"):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(args.n_batches):
        cur_t = diffusion.num_timesteps - args.skip_timesteps - 1

        samples = sample_fn(
            model,
            (args.batch_size, 3, side_y, side_x),
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=args.skip_timesteps,
            init_image=init,
            randomize_class=True,
        )

        for j, sample in enumerate(samples):
            if j % 100 == 0 or cur_t == 0:
                print()
                for image in sample["pred_xstart"]:
                    filename = output_path(args.prompts, args.image_prompts)
                    TF.to_pil_image(image.add(1).div(2).clamp(0, 1)).save(filename)
                    print(filename)
            cur_t -= 1


gc.collect()
do_run()
