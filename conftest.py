"""Make the ``src``-layout package importable when the suite is run from a bare
checkout, so the tests do not require ``pip install -e .`` first."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
