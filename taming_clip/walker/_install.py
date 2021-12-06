import sys
import urllib.request
from pathlib import Path
from typing import Dict

from rich import get_console, pretty, traceback
from tqdm import tqdm

ckpts = {
    "last.ckpt": "https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1",
    "model.yaml": "https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1",
}

console = get_console()
print = console.print
rule = console.rule

def wget(source: str, target: Path) -> None:
    with urllib.request.urlopen(source) as source, open(str(target), "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))


def update_path() -> None:
    for path in (Path(__file__).parents[1] / "lib").glob("*"):
        if path.is_dir():
            sys.path.insert(0, str(path))


def download_ckpts(ckpts: Dict) -> Path:
    cache_dir = Path.home() / ".cache/walker"
    cache_dir.mkdir(exist_ok=True, parents=True)

    for filename, url in ckpts.items():
        if not (cache_dir / filename).exists():
            wget(url, cache_dir / filename)
    return cache_dir


def install() -> Path:
    pretty.install()
    traceback.install()
    update_path()
    return download_ckpts(ckpts)
