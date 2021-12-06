from itertools import tee

import numpy as np
import pandas as pd
from scipy import stats


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def fftnoise(f):
    f = np.array(f, dtype='complex')
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1:Np+1] *= phases
    f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
    return np.fft.ifft(f).real

def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1/samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs>=min_freq, freqs<=max_freq))[0]
    f[idx] = 1
    return fftnoise(f)

def linspace_gaussian(start, stop, num=50):
    return stats.norm.ppf(np.linspace(stats.norm.cdf(start), stats.norm.cdf(stop), num))

def spline(a, b, N, p = 0.8):
    rN = int((1-p) * N)
    pi = np.linspace(a, a, N)
    pi[-rN:] = np.linspace(a, b, rN)
    return pi

def load_script(name: str) -> pd.DataFrame:
    return pd.read_csv(
        f"./scripts/{name}.csv", header=None, names=["line", "cue"]
    ).sort_values("cue")
