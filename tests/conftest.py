"""Test configuration.

Makes the ``src/`` layout importable when the suite is run without first
installing the package (e.g. a plain ``pytest`` from the repo root). When the
package *is* installed, the already-importable copy wins and this is a no-op.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))
