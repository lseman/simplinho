"""Make the locally built simplinho module importable from test modules."""

import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_requested = os.environ.get("SIMPLINHO_BUILD_DIR")
_build_dirs = ((_requested,) if _requested else
               ("build-local", "build-local/build", "build", "build-verify"))
for _d in _build_dirs:
    if not os.path.isabs(_d):
        _d = os.path.join(ROOT, _d)
    if glob.glob(os.path.join(_d, "simplinho*.so")):
        sys.path.insert(0, _d)
        break
