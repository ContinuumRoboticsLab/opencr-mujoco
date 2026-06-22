#!/usr/bin/env python3
"""Tests for the sysid_pipeline CLI conventions and optimizer progress display.

Locks two user-facing behaviors:
- --config resolves names from configs/sysid/ exactly like generate.py /
  teleop.py do, while explicit JSON paths keep working
- the parallel multistart's progress reporting (mm / elapsed formatting and
  the TTY-vs-plain status line) stays sane, since it replaced the raw
  per-iteration worker firehose as the default output
"""

import io
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sysid_pipeline import resolve_config_path, CONFIG_TYPE  # noqa: E402

SYSID_CONFIG_DIR = PROJECT_ROOT / "configs" / CONFIG_TYPE


class TestConfigResolution:
    def test_bare_name_resolves_into_configs_sysid(self):
        path = resolve_config_path("ftdcr_v4_pipeline")
        assert path == SYSID_CONFIG_DIR / "ftdcr_v4_pipeline.json"
        assert path.exists(), "shipped config must resolve"

    def test_json_path_passes_through(self):
        # Even a not-yet-existing .json path is treated as a path, so error
        # messages point at what the user typed.
        path = resolve_config_path("some/dir/custom.json")
        assert path == Path("some/dir/custom.json")

    def test_existing_file_passes_through(self, tmp_path):
        cfg = tmp_path / "saved_config"  # no .json suffix
        cfg.write_text("{}")
        assert resolve_config_path(str(cfg)) == cfg

    def test_all_shipped_configs_resolve_by_name(self):
        for cfg_file in SYSID_CONFIG_DIR.glob("*.json"):
            assert resolve_config_path(cfg_file.stem) == cfg_file


class TestCLI:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "sysid_pipeline.py", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def test_list_configs(self):
        result = self._run("--list-configs")
        assert result.returncode == 0, result.stderr
        assert "ftdcr_v4_pipeline" in result.stdout
        assert "ftdcr_v6_pipeline" in result.stdout

    def test_show_config_by_name(self):
        result = self._run("--config", "ftdcr_v4_pipeline", "--show-config")
        assert result.returncode == 0, result.stderr
        assert '"base_generation_config"' in result.stdout

    def test_unknown_config_lists_alternatives(self):
        result = self._run("--config", "does_not_exist")
        assert result.returncode != 0
        assert "ftdcr_v4_pipeline" in result.stdout

    def test_config_required(self):
        result = self._run()
        assert result.returncode != 0
        assert "--list-configs" in result.stderr


class TestProgressFormatting:
    def test_fmt_mm(self):
        from opencr_mujoco.sysid.parallel_optimizer import _fmt_mm

        assert _fmt_mm(0.003931) == "3.931 mm"
        assert _fmt_mm(float("inf")) == "--"  # no eval finished yet
        assert _fmt_mm(1e6) == "--"  # failed-start sentinel

    def test_fmt_elapsed(self):
        from opencr_mujoco.sysid.parallel_optimizer import _fmt_elapsed

        assert _fmt_elapsed(0) == "00:00"
        assert _fmt_elapsed(134) == "02:14"
        assert _fmt_elapsed(3600 + 125) == "1:02:05"

    def test_status_line_plain_stream_prints_periodically(self):
        from opencr_mujoco.sysid.parallel_optimizer import _StatusLine

        stream = io.StringIO()  # not a TTY
        status = _StatusLine(stream=stream, plain_period=1000.0)
        status.update("first")
        status.update("suppressed (inside period)")
        status.println("permanent")
        status.clear()
        out = stream.getvalue()
        assert out == "first\npermanent\n"
        assert "\r" not in out  # no terminal control codes off-TTY

    def test_first_update_prints_even_with_small_monotonic(self, monkeypatch):
        """time.monotonic() starts near zero on a freshly booted machine
        (e.g. a CI runner); the first status update must print regardless —
        a 0.0 'never printed' sentinel silently suppressed it."""
        from opencr_mujoco.sysid import parallel_optimizer

        monkeypatch.setattr(parallel_optimizer.time, "monotonic", lambda: 3.0)
        stream = io.StringIO()
        status = parallel_optimizer._StatusLine(stream=stream, plain_period=1000.0)
        status.update("first")
        assert stream.getvalue() == "first\n"

    def test_status_line_tty_repaints_in_place(self):
        from opencr_mujoco.sysid.parallel_optimizer import _StatusLine

        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        stream = FakeTTY()
        status = _StatusLine(stream=stream, tty_period=0.0)
        status.update("live 1")
        status.update("live 2")
        status.println("done line")
        out = stream.getvalue()
        # Live updates repaint via \r + erase, permanent line clears them first
        assert out.count("\r\x1b[2K") == 3
        assert out.endswith("done line\n")
