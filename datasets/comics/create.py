#%%
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image
import pytesseract
import re
import pandas as pd
from rich import print


# %%
def find_panels(path: Path, min_size: float = 0.01):
    orignal_img = cv.imread(str(path))
    height, width = orignal_img.shape[:-1]

    # Specific asterix fixed:
    orignal_img[-45:, :, :] = 255
    orignal_img[:, :10, :] = 255
    orignal_img[:, -5:, :] = 255
    #####

    img = cv.cvtColor(orignal_img, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(img,5)
    img = cv.filter2D(src=img, ddepth=-1, kernel=np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]]))
    _, img = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((9,9),np.uint8))
    _, _, img, _ = cv.floodFill(img,np.zeros((height+2, width+2), np.uint8),(0,0),255)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((9,9),np.uint8))

    contours, _ = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if  min_size * img.size < cv.contourArea(cnt) < 0.5 * img.size:
            x,y,w,h = cv.boundingRect(cnt)
            yield orignal_img[y:y+h, x:x+w]

def extract_panels(folder: Path, save_path: Path):
    print(f"Extracing panels from {folder}")
    for image in tqdm(list(folder.glob("*jpg"))):
        for i, panel in enumerate(find_panels(image)):
            path = save_path / f"{folder.name}_{image.stem}_{i:03}.jpg"
            cv.imwrite(str(path), panel) 


def image_to_text(path: Path, lang:str) -> str:
    text = pytesseract.image_to_string(Image.open(path), lang=lang)
    text = text.replace("\n", " ")
    text = re.sub('[^\w ]+','', text).lower().strip()
    return text

def folder_to_dataset(folder: Path, lang: str, include_empty: bool = True) -> pd.DataFrame:
    dataset = []
    print(f"Extracing text from {folder}")
    for image in tqdm(list(folder.glob("*jpg"))):
        text = image_to_text(image, lang=lang)
        if include_empty or text != "":
            dataset.append((image.name, text))
        if not include_empty and text == "":
            image.unlink()
    dataset = pd.DataFrame(dataset, columns=["name", "caption"])
    return dataset

def make_comic_dataset(folder: Path, lang: str):
    folder = Path(folder)
    save_path = Path(f"./{folder.name}_panel")
    save_path.mkdir(exist_ok=True)
    for archive in filter(Path.is_dir, folder.iterdir()):
        extract_panels(archive, save_path)
    dataset = folder_to_dataset(save_path, lang, False)
    dataset.to_csv(save_path.with_suffix(".csv"))

# %%
dataset = make_comic_dataset("./naruto", "rus")
# %%
# p = Path("/Users/mfr/Downloads/asterix/rus/")
# t = p / "asterix1to8"
# t.mkdir(exist_ok=True)
# for i in range(1, 9):
#     folder = p / f"asterix{i}"
#     for image in folder.glob("*jpg"):
#         name = f"a{i}_{image.stem}.jpg"
#         copyfile(image, t / name)
# %%
