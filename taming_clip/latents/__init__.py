"""isort:skip_file
"""
# from .sampler import Sampler, print

import rich
import rich.pretty
import rich.traceback
rich.pretty.install()
rich.traceback.install()
from rich.console import Console
console = Console()
del rich