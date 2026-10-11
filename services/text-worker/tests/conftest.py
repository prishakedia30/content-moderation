import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # service modules
sys.path.insert(0, str(HERE.parent.parent))   # services/ -> `common`
sys.path.insert(0, str(HERE))                 # fake_model
