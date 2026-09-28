"""Gate 15: the brain runs in a browser, spike for spike the same as brain/lif.py, and fast enough for five ducks.

Run: uv run python -m gates.gate_15_web     (needs Google Chrome; set CHROME if it is not in /Applications)

Packs the connectome with web/pack.py if web/data/ is missing, serves web/, and opens web/index.html in
headless Chrome. The page runs web/pack.py's reference schedule in a worker and posts back what it got
along with five brains' speed side by side. The gate asserts every tick's spike count and every neuron's
total match the Python run, and that the slowest of five brains keeps up with real time. Run by number
only, since it needs Chrome.
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
    if not (WEB / "data" / "reference.bin").exists():
        subprocess.run([sys.executable, "-m", "web.pack"], check=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/index.html?report"
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
    c, b = r["check"], r["bench"]
    print(f"reference: {c['ticks']} ticks, browser {c['spikes']:,} spikes, Python {c['refSpikes']:,}")
    if c["firstTick"] >= 0:
        print(f"  first differs at tick {c['firstTick']}; {c['neuronsOff']:,} neurons fire a different number of times")
    for i, d in enumerate(b["ducks"]):
        print(f"duck {i + 1}: {d['msPerTick']:.2f} ms a tick, {d['spikesPerTick']:.0f} spikes a tick")
    print(f"five brains: {b['speed']:.1f}x real time for the slowest")
    return verdict({
        "every tick fires as many neurons as in Python": c["firstTick"] < 0,
        "every neuron fires as often as in Python": c["neuronsOff"] == 0,
        f"five brains at once run at least {MIN_SPEED}x real time": b["speed"] >= MIN_SPEED,
    })


if __name__ == "__main__":
    sys.exit(main())
