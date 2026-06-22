#!/usr/bin/env python3
"""Run the opencr-mujoco smoke and unit-test suites.

Usage:
    python run_tests.py                 # all unit tests
    python run_tests.py --quick         # release smoke suite
    python run_tests.py -t generate     # one tests/unit/test_<name>.py module
    python run_tests.py -v              # verbose pytest output
"""

import sys
import subprocess
from pathlib import Path
import argparse
from importlib.util import find_spec


def run_tests(verbose=False, specific_test=None):
    """Run unit tests for the project.

    Args:
        verbose: Show detailed test output
        specific_test: Run only a specific test module

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Get test directory
    test_dir = Path(__file__).parent / "tests" / "unit"

    if specific_test:
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]

        if verbose:
            cmd.append("-v")

        # Run specific test
        test_file = test_dir / f"test_{specific_test}.py"
        if not test_file.exists():
            print(f"Error: Test file {test_file} not found")
            return 1
        cmd.append(str(test_file))
    else:
        cmd = [sys.executable, "-m", "pytest", str(test_dir)]

        if verbose:
            cmd.append("-v")

    # Add coverage if available
    if find_spec("pytest_cov") is not None:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])

    # Add other useful pytest options
    cmd.extend(
        [
            "-x",  # Stop on first failure
            "--tb=short",  # Shorter traceback format
            "--color=yes",  # Colored output
        ]
    )

    print(f"Running tests: {' '.join(cmd)}")
    print("-" * 60)

    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    return result.returncode


def run_pytest_targets(targets, verbose=False):
    """Run pytest against explicit files/directories."""
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    cmd.extend(str(Path(__file__).parent / target) for target in targets)
    cmd.extend(
        [
            "-x",
            "--tb=short",
            "--color=yes",
        ]
    )

    print(f"Running tests: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def _available_test_modules():
    """Discover test modules from tests/unit/test_*.py."""
    test_dir = Path(__file__).parent / "tests" / "unit"
    return sorted(p.stem[len("test_") :] for p in test_dir.glob("test_*.py"))


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Run unit tests for opencr-mujoco")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed test output"
    )
    parser.add_argument(
        "-t",
        "--test",
        choices=_available_test_modules(),
        help="Run only a specific test module",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick smoke tests only"
    )

    args = parser.parse_args()

    if args.quick:
        print("Running quick smoke tests...")
        modules = [
            "generate",
            "viewer",
            "teleop",
            "evaluate",
            "evaluation",
            "pretension",
            "scene_creation",
        ]
        for module in modules:
            print(f"\nTesting {module}...")
            ret = run_tests(verbose=False, specific_test=module)
            if ret != 0:
                print(f"Failed on {module} tests")
                return ret

        print("\nTesting main scripts...")
        ret = run_pytest_targets(["tests/test_main_scripts.py"], verbose=False)
        if ret != 0:
            print("Failed on main script tests")
            return ret

        print("\n✅ All quick tests passed!")
        return 0

    # Run tests
    ret = run_tests(verbose=args.verbose, specific_test=args.test)

    if ret == 0:
        print("\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ Tests failed with exit code {ret}")

    return ret


if __name__ == "__main__":
    sys.exit(main())
