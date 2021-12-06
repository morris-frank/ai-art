import clip
from torch import Tensor


class Perceptor:
    def __init__(self, device: str) -> None:
        self.device = device
        self.model = self.load()

    def load(self) -> clip.model.CLIP:
        model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        return model.eval()

    def txt2embedding(self, text: str) -> Tensor:
        text = clip.tokenize(text).to(self.device)
        return self.model.encode_text(text).detach().clone().cpu()

    def img2embedding(self, x: Tensor) -> Tensor:
        return self.model.encode_image(x)
