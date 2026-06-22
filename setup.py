#!/usr/bin/env python3
"""Legacy shim — all project metadata lives in pyproject.toml.

Kept only so very old pip versions (< 21.3) can still do editable installs;
modern pip ignores this file and uses the PEP 517/660 path.
"""

from setuptools import setup

setup()
