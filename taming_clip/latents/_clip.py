import torch
from pathlib import Path
from torch.nn import functional as F
from torch import nn
from typing import Union
import clip
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image


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


class CLIP:
    def __init__(self, device: str, cutn=16) -> None:
        self.device = device
        model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        self.model = model.eval().requires_grad_(False)
        self.size = self.model.visual.input_resolution
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.make_cutouts = MakeCutouts(self.size, cutn)

    def parse_prompt(string: str) -> Union[str, Path]:
        return Path(string) if Path(string).exists() else string

    def prompts2embeddings(self, prompts: list[str]) -> Tensor:
        embeddings = []
        for prompt in prompts:
            if Path(prompt).exists():
                embeddings.append(self.img2embedding(prompt))
                pass
            else:
                embeddings.append(self.txt2embedding(prompt))
        embedding = torch.cat(embeddings).float().to(self.device)
        return embedding

    def txt2embedding(self, text: str) -> Tensor:
        text = clip.tokenize(text).to(self.device)
        return self.model.encode_text(text).detach().clone().cpu()

    def img2embedding(self, path: Path) -> Tensor:
        img = Image.open(path).convert("RGB")
        img = TF.resize(img, min(side_x, side_y, *img.size), transforms.InterpolationMode.LANCZOS)
        batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
        return self.model.encode_image(self.normalize(batch)).float()
