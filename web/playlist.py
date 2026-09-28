"""Fetches the music box's Spotify playlist and writes its songs into web/view/tracks.js, for the box to shuffle.

Run: uv run python -m web.playlist     (again whenever songs are added to or taken off the playlist)

Spotify's embedded player cannot shuffle or start partway into a playlist, so the box plays the songs one at a time
in its own shuffled order. This reads the song list from the playlist's public embed page; it stores only Spotify's
ids for the songs, and the music itself stays with Spotify.
"""
import json
import re
import urllib.request
from pathlib import Path

PLAYLIST = "2vg04PEDQePvRsS7PD3TyS"  # web/view/spotify.js has the same
OUT = Path(__file__).resolve().parent / "view" / "tracks.js"


def tracks(playlist: str) -> list[dict]:
    page = urllib.request.urlopen(urllib.request.Request(f"https://open.spotify.com/embed/playlist/{playlist}",
                                                         headers={"User-Agent": "Mozilla/5.0"})).read().decode()
    data = json.loads(re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', page, re.S)[1])

    def find(node):
        if isinstance(node, dict):
            if "trackList" in node:
                return node["trackList"]
            node = list(node.values())
        if isinstance(node, list):
            for child in node:
                found = find(child)
                if found is not None:
                    return found
        return None

    return [{"uri": t["uri"], "ms": t.get("duration", 0)} for t in find(data) or [] if t.get("uri", "").startswith("spotify:track:")]


def main() -> None:
    songs = tracks(PLAYLIST)
    if not songs:
        raise SystemExit("found no songs on the playlist's embed page; Spotify may have changed it")
    OUT.write_text("// The music box's playlist, song by song, written by web/playlist.py. Spotify's ids only.\n"
                   f"export const PLAYLIST = {json.dumps(PLAYLIST)};\n"
                   f"export const TRACKS = {json.dumps([s['uri'] for s in songs])};\n")
    print(f"{len(songs)} songs written to {OUT}")


if __name__ == "__main__":
    main()
