"""Read-only reuse of audited sampling/allocation and observed-panel loaders."""
import sys
from pathlib import Path
PREVIOUS=Path(__file__).resolve().parents[1]/'root_updates'
sys.path.insert(0,str(PREVIOUS))
import mechanisms as old
import data as old_data
original=old.original
