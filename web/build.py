"""Gathers the browser garden into one folder to upload to a website: web/dist/garden/, served at /garden/.

Run: uv run python -m web.build     (after web.pack)

Everything the garden page loads and nothing else: the page, its scripts, the packed brain and eyes, the duck's
rig and motions, a cover picture, and the licences. The check page and its reference data stay behind.
"""
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
OUT = WEB / "dist" / "garden"
SCRIPTS = ["bundle.js", "brain.js", "lif.js", "vision.js", "worker.js"]
DATA = ["brain.bin.gz", "eyes.bin.gz", "robot.json", "clips.json", "LICENSE-microduck"]


def main() -> None:
    missing = [name for name in DATA if not (WEB / "data" / name).exists()]
    if missing:
        raise SystemExit(f"run `uv run python -m web.pack` first; missing {', '.join(missing)}")
    shutil.rmtree(OUT, ignore_errors=True)
    (OUT / "data").mkdir(parents=True)
    shutil.copy(WEB / "index.html", OUT)
    for name in SCRIPTS:
        shutil.copy(WEB / name, OUT)
    for folder in ("garden", "view"):
        shutil.copytree(WEB / folder, OUT / folder)
    for name in DATA:
        shutil.copy(WEB / "data" / name, OUT / "data")
    shutil.copy(ROOT / "docs" / "garden.webp", OUT / "cover.webp")
    shutil.copy(ROOT / "LICENSE", OUT / "LICENSE")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}: {sum(1 for p in OUT.rglob('*') if p.is_file())} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
