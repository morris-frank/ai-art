import io
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests
from rich import get_console, pretty
import collections.abc
from rich import print
from rich import traceback
import re
from argparse import ArgumentParser as _ArgumentParser

console = get_console()
pretty.install()
traceback.install()

CWD = Path(__file__).parent
input_path = CWD.parent / "inputs"
results_path = CWD / "results"

def ArgumentParser() -> _ArgumentParser:
    parser = _ArgumentParser()
    parser.add_argument("-d", "-device",   dest="device", default="cuda:0")
    parser.add_argument("-c", "-continue", dest="cont", action="store_true", help="Do not delete previous results.")
    weight_dirs = list(filter(lambda d: (CWD/d).exists(), ["models", "weights", "checkpoints"]))
    # if len(weight_dirs) > 0:
    #     parser.add_argument("-chkpt", "-weights", "-checkpoint", dest="checkpoint")
    if (CWD / "checkpoints").exists():
        print("checkpoints exists")
    return parser

def file_input(path: str):
    if path is None:
        return None
    if str(path).startswith("http://") or str(path).startswith("https://"):
        r = requests.get(path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    else:
        if len(list(input_path.glob(f"{path}*"))) == 1:
            path = next(input_path.glob(f"{path}*"))
        return open(path, "rb")

def output_path(*names: str, ext="png") -> Path:
    def tostr(obj: Any) -> str:
        return ".".join(map(tostr, obj)) if isinstance(obj, collections.abc.Sequence) and not isinstance(obj, str) else re.sub('[^\w ]+','', str(obj)).lower().strip().replace(" ", ".")
    folder = results_path / tostr(names)
    folder.mkdir(exist_ok=True, parents=True)
    i = 0
    while (folder / f"{i:05}.{ext}").exists():
        i += 1
    return folder / f"{i:05}.{ext}"


@contextmanager
def log(action: str):
    start_time = time.perf_counter()
    print(f"[yellow]{action}…[/yellow]")
    try:
        yield None
    finally:
        duration = time.perf_counter() - start_time
        print(f"[green]{action} ✔[/green] ({int(duration)}s)")
