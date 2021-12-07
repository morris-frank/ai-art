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

console = get_console()
pretty.install()
traceback.install()

input_path = Path(__file__).parents[1] / "inputs"
results_path = Path(__file__).parent / "results"

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
