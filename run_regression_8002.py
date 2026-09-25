# Thin convenience wrapper: runs the regression suite against the
# port-8005 deployment without having to `cd chat/backend/tests` first.
#
# Fixed: this used to do `rs.API_BASE = "..."` AFTER importing
# chat.backend.tests.regression_suite, which by then had already failed
# at import time with `ModuleNotFoundError: No module named 'bench_config'`
# (bench_config.py is a sibling of regression_suite.py, so importing the
# suite as a dotted package from the repo root, with the repo root as the
# only thing on sys.path, could never find it). Fixed by putting
# chat/backend/tests on sys.path AND setting FQC_API_BASE (the env var
# bench_config actually reads) before the import happens.
import os
import sys

_TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat", "backend", "tests")
os.environ.setdefault("FQC_API_BASE", "http://127.0.0.1:8005")
sys.path.insert(0, _TESTS_DIR)

import chat.backend.tests.regression_suite as rs

rs.main()
