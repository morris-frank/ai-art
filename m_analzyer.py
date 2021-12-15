from pathlib import Path
import librosa
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("file", type=str, help="music file")
parser.add_argument("-fps", type=int, default=25)
parser.add_argument("-beat_measure", type=int, default=16, help="Measure of the song, counted in beats")
parser.add_argument("-beat_phase", type=int, default=32, help="Start of the first measure, counted in beats")


cli = parser.parse_args()



wave, sr = librosa.load(f"inputs/{cli.file}")

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

def compute_beat_markers(wave, sr, osize):
    _, beat_idx = librosa.beat.beat_track(y=wave, sr=sr)
    beat_idx = np.round(librosa.frames_to_time(beat_idx, sr=sr) * cli.fps).astype(int)
    beats = np.zeros(osize)
    beats[beat_idx[cli.beat_phase :: cli.beat_measure]] = 1
    beats = np.convolve(beats, [0, 0, 0, 0, 0, 0, 0, 1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01], mode="same")
    return beats

eq = compute_eq(wave, sr, amax=35, eq_bins=16)
beats = compute_beat_markers(wave, sr, eq.shape[1])

Path("frames").mkdir(exist_ok=True)
imsize = 256
qrts = imsize//4
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)
bcnt = 0
for i, (b, eq) in enumerate(tqdm(zip(beats, eq.T), total=beats.size)):
    im = np.zeros((imsize, imsize, 3), dtype=np.uint8)
    im[..., :] = np.array([247, 247, 247])
    im[imsize-int(imsize*b):imsize, :, :] = eq[0] * np.array([116, 162, 93]) + (1-eq[0]) * np.array([247, 247, 247])
    im = Image.fromarray(im)
    ImageDraw.Draw(im).text((0, 0), f":{bcnt}", font=fnt, fill=(0, 0, 0))
    if b == 1:
        bcnt += 1
    im.save(f"./frames/{i:05d}.png")
print("[red]Start ffmpeg[/red]")
outpath = Path(cli.file).with_suffix(".mp4")
subprocess.call(["ffmpeg", "-i", "frames/%05d.png", "-i", f"inputs/{cli.file}", "-c:v", "libx264", outpath])
print("[red]Finished ffmpeg[/red]")