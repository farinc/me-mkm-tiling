#!/usr/bin/env python
"""Regenerate python/me_mkm/_me_mkm/__init__.pyi from the Rust sources.
Runs `cargo run --bin stub_gen`
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

env = os.environ.copy()
# Prepend the interpreter dir and its base prefix (where python3xx.dll lives on
# Windows) so the linked libpython is discoverable. Harmless elsewhere.
dll_dirs = [os.path.dirname(sys.executable), sys.base_prefix]
env["PATH"] = os.pathsep.join(dll_dirs + [env.get("PATH", "")])

proc = subprocess.run(
    ["cargo", "run", "--quiet", "--bin", "stub_gen"], cwd=ROOT, env=env
)
sys.exit(proc.returncode)
