"""Gate 15: the browser garden is the Python garden, and a computer can run five ducks of it at real time.

Run: uv run python -m gates.gate_15_web     (needs Google Chrome; set CHROME if it is not in /Applications)

Packs the garden with web/pack.py if web/data/ is missing, serves web/, and opens web/check.html in headless
Chrome. The page runs web/check.js, which puts the same inputs through the Python modules (recorded by
web/pack.py) and their JavaScript ports, then times five brains with their eyes open, each in its own worker.
The gate asserts every check passes: the brain spike for spike with learning, the eyes within tolerance, the
body, world and readouts to rounding; and that the slowest of five ducks keeps up with real time. Run by
number only, since it needs Chrome.
"""
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
from functools import partial
from pathlib import Path

from gates.episodes import verdict

CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
WEB = Path(__file__).resolve().parent.parent / "web"
MIN_SPEED = 1.0  # x real time for the slowest of five brains; the body and the world still need room
TIMEOUT_S = 300


class Handler(http.server.SimpleHTTPRequestHandler):
    result: dict | None = None
    done = threading.Event()

    def do_POST(self):
        Handler.result = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(204)
        self.end_headers()
        Handler.done.set()

    def log_message(self, *args):
        pass


def main() -> int:
    if not (WEB / "data" / "reference.bin.gz").exists():
        subprocess.run([sys.executable, "-m", "web.pack"], check=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/check.html?report"
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen([CHROME, "--headless=new", f"--user-data-dir={profile}", "--no-first-run",
                                   "--remote-debugging-port=0", url],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            Handler.done.wait(TIMEOUT_S)
        finally:
            chrome.terminate()
            chrome.wait()
            server.shutdown()
    r = Handler.result
    if r is None or "error" in r:
        print("the page did not report" if r is None else f"the page failed: {r['error']}")
        print("FAIL")
        return 1
    rows, b = r["rows"], r["bench"]
    for row in rows:
        print(f"  {'ok  ' if row['ok'] else 'FAIL'} {row['name']}: {row['detail']}")
    for i, d in enumerate(b["ducks"]):
        print(f"duck {i + 1}: {d['msPerStep']:.1f} ms a 20 ms step, {d['spikesPerTick']:.0f} spikes a tick")
    print(f"five brains: {b['speed']:.1f}x real time for the slowest")
    return verdict({
        f"all {len(rows)} checks against the Python garden pass": all(row["ok"] for row in rows),
        f"five ducks with eyes run at least {MIN_SPEED}x real time": b["speed"] >= MIN_SPEED,
    })


if __name__ == "__main__":
    sys.exit(main())
