import sys
from pathlib import Path
import warnings

# Ensure the src directory is on the path so pfd_toolkit can be imported
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Suppress DeprecationWarnings from by pymupdf's SWIG wrappers
# ...we don't call these objects, so the warnings are benign
warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPyPacked|swigvarlink|SwigPyObject) has no __module__ attribute",
    category=DeprecationWarning,
)