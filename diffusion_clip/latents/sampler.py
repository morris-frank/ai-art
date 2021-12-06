import os
from itertools import product
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from . import print, rule
from .generator import Generator
from .clip import CLIP


class Pivot(torch.nn.Module):
    def __init__(self, width, height, batch_size=1):
        super().__init__()
        self.embedding = (
            0.5 * torch.randn(batch_size, 256, width // 16, height // 16).cuda()
        )
        self.embedding = torch.nn.Parameter(
            torch.sinh(1.9 * torch.arcsinh(self.embedding))
        )

    def forward(self):
        return self.embedding.clip(-6, 6)


class Sampler:
    target_names = list(
        map("_".join, product(("generation", "perception"), ("attractor", "repeller")))
    )

    def __init__(
        self,
        device: str,
        width: int = 512,
        height: int = 512,
        targets: Dict[str, Any] = {},
    ) -> None:
        self.width, self.height = width, height
        self.device = device

        self.generator = Generator(width, height, device)
        self.perceptor = CLIP(device)

        self.weight_decay = 0.1
        self.batch_size = 1
        self.lr = 0.5

        for name in self.target_names:
            setattr(self, name, None)
            setattr(self, f"_{name}", targets.get(name, None))
            setattr(self, f"_{name}s", [])
        rule("Finished set-up Sampler().")

    def reset(self):
        self.pivot = Pivot(self.width, self.height, self.batch_size).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [{"params": [self.pivot.embedding], "lr": self.lr}],
            weight_decay=self.weight_decay,
        )

    def embed(self, embed_or_string: Any, target: str):
        if not isinstance(embed_or_string, str):
            return embed_or_string.to(self.device)
        else:
            if target.startswith("generation"):
                return self.generator.img2embedding(embed_or_string).to(self.device)
            if target.startswith("perception"):
                return self.perceptor.txt2embedding(embed_or_string).to(self.device)
            else:
                raise ValueError

    def build_loss_targets(self):
        for name in self.target_names:
            user_set, embeddings = getattr(self, f"_{name}"), getattr(self, f"_{name}s")
            if user_set is not None:
                embeddings = embeddings + [user_set]
            if len(embeddings) > 0:
                comb = sum([self.embed(embedding, name) for embedding in embeddings])
                setattr(self, name, comb / comb.norm(dim=-1, keepdim=True))
            else:
                setattr(self, name, None)

    def imsave(self, path: str, **kwargs):
        self.generator.imsave(self.pivot(), path, **kwargs)

    def step(self) -> Tensor:
        image_at_pivot = self.generator.embedding2img_augmented(self.pivot())
        perception_at_pivot = self.perceptor.img2embedding(image_at_pivot)

        def _loss_element(target, preds, factor):
            if target is not None:
                return factor * torch.cosine_similarity(target, preds).mean()
            else:
                return 0

        loss = (
            _loss_element(self.perception_attractor, perception_at_pivot, -10)
            + _loss_element(self.perception_repeller, perception_at_pivot, +5)
            + _loss_element(self.generation_attractor, self.pivot(), -10)
            + _loss_element(self.generation_repeller, self.pivot(), +1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        use_weight_decay = torch.abs(self.pivot()).max() > 5
        for g in self.optimizer.param_groups:
            g["weight_decay"] = self.weight_decay if use_weight_decay else 0

        return loss

    def train(self, img_name: str, n_steps: int):
        self.build_loss_targets()

        for i in range(n_steps + 1):
            loss = self.step().item()
            if i % 500 == 0 or i == n_steps:
                for g in self.optimizer.param_groups:
                    print(
                        f"step={i:05d}, loss={loss:.3e}, lr={g['lr']}, decay={g['weight_decay']}"
                    )
                self.imsave(f"keyframes/{img_name}", save_embed=(i == n_steps))

    def generate_keyframes_v4(
        self, script: pd.DataFrame, n_steps_per_line: int, continuous: bool
    ):
        print(f"Start sampling\t{os.getcwd()}")
        self.reset()
        for index, line in script.iterrows():
            if not continuous:
                self.reset()
            name = f"{index:05d}"
            rule(name)
            print(f'Text\t "{line.line}"')

            self._perception_attractors = [line.line]
            self.train(name, n_steps_per_line)

    def generate_keyframes_v3(
        self,
        song: pd.DataFrame,
        fps: int,
        iter_per_frame: int = 100,
        warm_up_factor: float = 1.1,
    ):
        self.reset()
        song["embedding"] = song.line.apply(self.perceptor.txt2embedding)

        assigs = []
        starting = True
        for win in song.rolling(3, center=True):
            cues = win.cue.tolist()
            indx = win.index.tolist()
            _i = len(win) == 2
            win_size = int((cues[2 - _i] - cues[1 - _i]) * fps)
            if _i:
                if starting:
                    indx.insert(0, None)
                    starting = False
                else:
                    indx.append(None)
            for i in range(win_size):
                assigs.append((indx, i / win_size))

        for frame, ((i_left, i_mid, i_right), progress) in enumerate(assigs):
            name = f"{frame:05d}"
            print(name)

            if (keyframe := self.path["keyframes"] / f"{name}.p").exists():
                latent = torch.load(keyframe)
                self.pivot.embedding.data = latent.to(self.device)
                continue

            left, mid, right = None, song.loc[i_mid], None
            if i_left is not None:
                left = song.loc[i_left]
            if i_right is not None:
                right = song.loc[i_right]

            self._perception_attractors = [mid.embedding]
            if right is not None and progress >= 0.8:
                self._perception_attractors.append(right.embedding)

            self._perception_repellers = []
            if left is not None and progress <= 0.2:
                self._perception_repellers.append(left.embedding)

            warm_up = warm_up_factor * (1 - np.sqrt(frame / len(assigs)))
            iters = int(iter_per_frame + warm_up * iter_per_frame)
            self.train(name, iters)
