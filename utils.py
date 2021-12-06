import time
from contextlib import contextmanager

from rich import get_console, pretty, print, traceback

console = get_console()
pretty.install()
traceback.install()


@contextmanager
def log(action: str):
    start_time = time.perf_counter()
    print(f"[yellow]Started {action}[/yellow]")
    try:
        yield None
    finally:
        duration = time.perf_counter() - start_time
        print(f"[green]Finished {action}[/green] ({int(duration)}s)")
