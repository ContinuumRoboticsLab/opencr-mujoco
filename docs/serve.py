#!/usr/bin/env python3
"""No-cache local dev server for the demo site.

``python -m http.server`` sends no ``Cache-Control`` header, so browsers
heuristically cache scene XML / meshes / JS / the manifest and keep serving a
stale copy after you rebuild (e.g. a freshly-added prop never shows up, or an
edited controller doesn't take effect) until the heuristic window expires.

This serves ``docs/`` with ``Cache-Control: no-store`` on every response, so a
plain browser reload always gets the current build. For local development only;
GitHub Pages sets its own caching headers in production.

    python docs/serve.py            # http://localhost:8000
    python docs/serve.py 8080       # custom port
"""

import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(Path(__file__).resolve().parent)  # serve docs/ wherever invoked from
    httpd = ThreadingHTTPServer(("", port), _NoCacheHandler)
    print(f"Serving docs/ (no-cache) at http://localhost:{port}  —  Ctrl-C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
