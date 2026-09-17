"""Runner de tests: ejecuta toda la suite OFFLINE sin red ni créditos.

Uso:
    venv/bin/python -m unittest discover -s tests -p "test_*.py" -v
"""
from __future__ import annotations

import unittest


def run_all() -> None:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    run_all()