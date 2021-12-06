from pathlib import Path
from tqdm import tqdm

from ISR.models import RDN
import numpy as np
from PIL import Image

def upscale():
    out = Path("./upscaled_frames")
    out.mkdir(exist_ok=True)

    rdn = RDN(weights='psnr-small')
    for img_path in tqdm(Path("interpolations").glob("*png")):
        img = Image.open(img_path)
        lr_img = np.array(img)
        sr_img = rdn.predict(lr_img)
        img = Image.fromarray(sr_img)
        img.save(out / img_path.name)