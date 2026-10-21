"""Project-local inputs and isolated outputs for the screen-retention analysis."""
from pathlib import Path
import os
import sys

ANALYSIS = Path(__file__).resolve().parents[2]
ROOT = ANALYSIS.parent
RAW = Path(os.environ.get('M2_RAW_DIR', ANALYSIS / 'data' / 'raw')).resolve()
OUT = Path(os.environ.get('M2_OUTPUT_DIR', ANALYSIS / 'out' / 'm2')).resolve()
SCOPE = os.environ.get('M2_SCOPE', 'cleaned')
if SCOPE != 'cleaned':
    raise ValueError(f'Invalid analysis scope: {SCOPE!r}')
sys.path.insert(0, str(ANALYSIS / 'scripts'))
from cleanup import apply_cleanup, assert_clean
OUT.mkdir(parents=True, exist_ok=True)


def active_variant(variant=None):
    """Explicit variants take precedence over the process's isolated analysis route."""
    name = variant if variant is not None else os.environ.get('M2_VARIANT', 'clfzero')
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in name):
        raise ValueError(f'Invalid analysis variant: {name!r}')
    return name


def data_dir(variant=None):
    """Return the chosen data domain without redirecting an explicit raw request."""
    variant = active_variant(variant)
    d = out_dir(variant) / 'processed'
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(variant=None):
    variant = active_variant(variant)
    d = OUT if variant == 'clfzero' else OUT / variant
    d.mkdir(parents=True, exist_ok=True)
    return d
