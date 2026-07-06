"""
Always rebuilds and reinstalls the Rust extension (me_mkm._me_mkm) before the
test session starts, so editing src/*.rs and then running pytest -- whether
from a terminal or VSCode's Test Explorer -- always exercises the current
Rust code, not a stale compiled .so left over from the last manual
`maturin develop`.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config):
    subprocess.run(
        ["maturin", "develop", "--release"],
        cwd=REPO_ROOT,
        check=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
