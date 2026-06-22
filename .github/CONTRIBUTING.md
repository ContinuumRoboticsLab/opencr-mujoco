# Contributing to opencr-mujoco

Thanks for helping improve `opencr-mujoco`. This repository is an academic
research codebase, so the most valuable contributions are clear, reproducible,
and conservative: small fixes, tests that lock down behavior, documentation
that helps another researcher reproduce results, and focused new examples.

## Ways to contribute

- Report bugs with the exact command, config name or file, OS, Python version,
  MuJoCo version, and traceback.
- Improve documentation, examples, or configuration comments when something is
  hard to reproduce.
- Add tests for controller behavior, TDCR generation, evaluation conventions,
  or system-identification data handling.
- Propose new robot configurations or scenes through JSON configs rather than
  committing generated XMLs.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

On macOS, GUI teleoperation and passive MuJoCo viewer workflows should be run
with `mjpython`. Headless commands, config listing, tests, and generation work
with ordinary `python`.

## Checks before a pull request

```bash
black .
flake8 . --count --statistics
python run_tests.py --quick
```

For changes that touch the evaluation pipeline or generator physics, also run
the relevant correctness tests:

```bash
pytest tests/unit/test_generator_physics.py tests/unit/test_evaluation_correctness.py -q
```

## Repository hygiene

- Do not commit generated `assets/tdcr/*.xml` files; they regenerate from
  `configs/generation/`.
- Keep `docs/scenes/` XML files tracked when the browser demo manifest marks
  them available.
- Do not commit large website videos. Upload them as GitHub Release assets and
  link to the release URL from `docs/index.html`.
- Do not commit local editor, environment, or coding-agent files.

## Pull request style

Keep PRs narrow when possible. A good PR description says what changed, why it
changed, how it was tested, and whether it affects published evaluation
conventions or reference data.
