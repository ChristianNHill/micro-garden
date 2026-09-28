"""Gathers the browser garden into one folder to upload to a website: web/dist/garden/, served at /garden/.

Run: uv run python -m web.build     (after web.pack)

Everything the garden page loads and nothing else: the page, its scripts, the packed brain and eyes, the duck's
rig and motions, a cover picture, and the licences. The check page and its reference data stay behind.
"""
import hashlib
import re
import shutil
from pathlib import Path

WEB = Path(__file__).resolve().parent
ROOT = WEB.parent
OUT = WEB / "dist" / "garden"
SCRIPTS = ["bundle.js", "brain.js", "lif.js", "vision.js", "worker.js"]
DATA = ["brain.bin.gz", "eyes.bin.gz", "robot.json", "clips.json", "LICENSE-microduck"]


SCRIPT = re.compile(r"""((?:from|import)\s*\(?\s*|new URL\(\s*|src=)(["'])(\.{0,2}/?[\w./-]+\.js)\2""")
DATA_URL = re.compile(r"""(["'])(data/[\w.-]+)\1""")


def stamp(out: Path) -> None:
    """Put a version on every script and data address. Web hosts tell browsers to keep these files for a month, so
    without it a returning visitor would run last month's garden. Each file's version comes from its own content and
    its imports' versions, so a change gives new addresses to that file and to what imports it, and nothing else."""
    pages = list(out.rglob("*.js")) + [out / "index.html"]
    source = {p: p.read_text() for p in pages}
    data = {f"data/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()[:10] for p in (out / "data").iterdir()}
    done: dict[Path, str] = {}

    def target(p: Path, rel: str) -> Path:
        return (p.parent / rel).resolve() if rel.startswith(".") else (out / rel).resolve()

    def rewrite(p: Path) -> str:
        text = SCRIPT.sub(lambda m: f"{m[1]}{m[2]}{m[3]}?v={version(target(p, m[3]))}{m[2]}", source[p])
        return DATA_URL.sub(lambda m: f"{m[1]}{m[2]}?v={data[m[2]]}{m[1]}", text)

    def version(p: Path) -> str:
        if p not in done:
            done[p] = "…"  # a file importing itself round a cycle keeps a placeholder
            done[p] = hashlib.sha256(rewrite(p).encode()).hexdigest()[:10]
        return done[p]

    for p in pages:
        version(p)
    for p in pages:
        p.write_text(rewrite(p))


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
    stamp(OUT)
    # for the portfolio pages on the same site: the hover card and the moving picture
    shutil.copy(WEB / "site" / "card.js", OUT / "card.js")
    shutil.copy(WEB / "site" / "cover.gif", OUT / "cover.gif")  # recorded from this garden with ?record
    shutil.copy(ROOT / "LICENSE", OUT / "LICENSE")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{OUT}: {sum(1 for p in OUT.rglob('*') if p.is_file())} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
