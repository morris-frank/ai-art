from pathlib import Path
from rudalle import get_realesrgan
from rudalle.pipelines import super_resolution
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

device = "cuda:0"
realesrgan = get_realesrgan("x2", device=device)
batch_size=4

Path("./upsampled_frames").mkdir(exist_ok=True)
with torch.inference_mode():
    for fp in tqdm(list(Path("./frames").glob("*png"))):
        pil_image = Image.open(fp).convert("RGB")
        sr_image = realesrgan.predict(np.array(pil_image), batch_size=batch_size)
        Image.fromarray(sr_image).save(f"./upsampled_frames/{fp.name}")

