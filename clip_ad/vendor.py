"""Import DAPO's vendored OpenCLIP, including its absolute internal imports."""

import importlib
from pathlib import Path
import sys


DAPO_ROOT = Path(__file__).resolve().parents[1] / 'DAPO'


def open_clip():
    existing = sys.modules.get('open_clip')
    if existing is not None:
        if Path(existing.__file__).resolve().parent != DAPO_ROOT / 'open_clip':
            raise RuntimeError('clip_ad requires DAPO/open_clip; another open_clip is already imported')
        return existing
    # DAPO's OpenCLIP has absolute imports, so importing it as DAPO.open_clip
    # would create duplicate module trees. Select the vendored package explicitly.
    sys.path.insert(0, str(DAPO_ROOT))
    return importlib.import_module('open_clip')
