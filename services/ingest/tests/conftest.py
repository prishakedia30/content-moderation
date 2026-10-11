import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
os.environ.setdefault("STORAGE", "local")
os.environ.setdefault("LOCAL_STORAGE_DIR", tempfile.mkdtemp())
os.environ.setdefault("DB_INIT", "0")
