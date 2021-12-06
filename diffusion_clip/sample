#! /usr/bin/env python
import os
from argparse import ArgumentParser
from pathlib import Path

from walker import Sampler, print
from walker.utils import load_script


def prepare_result_folder(name):
    i = 0
    while (results_path := Path(f"./results/{name}_{i:03d}")).exists():
        i += 1
    results_path.mkdir()
    os.chdir(results_path)


def main(cfg):
    if (Path("./results") / cfg.script).exists():
        print("Found results folder exists already, will continue sampling.")
        script = load_script("_".join(cfg.script.split("_")[:-1]))
        os.chdir(f"./results/{cfg.script}")
    else:
        script = load_script(cfg.script)
        prepare_result_folder(cfg.script)

    walker = Sampler(
        device=cfg.device,
        width=cfg.width,
        height=cfg.height,
        targets={
            "perception_attractor": cfg.txt,
            "perception_repeller": "disconnected, confusing, incoherent",
            "generation_attractor": cfg.img
        }
    )
    walker.generate_keyframes_v4(script, continuous=cfg.continuous, n_steps_per_line=cfg.n_steps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(dest="script", type=str, help="Name of the instruction set script to sample according to.", choices=list(map(lambda x: x.stem, Path("./scripts").glob("*csv"))))

    parser.add_argument("-device", type=str, help="Device to work on e.g. \"cuda:1\"", required=True)
    parser.add_argument("-img", type=str, help="Image style file to use as an attractor. Loaded from './img/'", default=None)
    parser.add_argument("-txt", type=str, help="String to use as additional embedding attractor in each step.", default=None)
    parser.add_argument("-c", "--continuous", action="store_true", help="Whether to sample continously or restart at each keyframe.", dest="continuous")
    parser.add_argument("-n", "--n_steps", type=int, help="Number of steps per sampling iterartion.", default=5_000)

    parser.add_argument("-width", type=int, default=512)
    parser.add_argument("-height", type=int, default=512)
    main(parser.parse_args())
