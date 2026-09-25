# Thin convenience wrapper: runs the regression suite against a backend
# on port 8001, e.g. a second instance started for comparison. See
# run_regression_8005.py for why this needs the sys.path insert and the
# env var (not a post-import attribute assignment) to actually work.
import os
import sys

_TESTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat", "backend", "tests")
os.environ.setdefault("FQC_API_BASE", "http://127.0.0.1:8001")
sys.path.insert(0, _TESTS_DIR)

import chat.backend.tests.regression_suite as rs

rs.main()
