// The music box's playlist, in Spotify's own embedded player, shown while the box is in the garden and paused when it
// is picked up. Spotify's player cannot shuffle or start partway into a playlist, so the box shuffles the songs itself
// (web/view/tracks.js, written by web/playlist.py) in a new order each visit and plays them one at a time, loading the
// next as each ends. A visitor signed in to Spotify in this browser hears whole songs; anyone else hears Spotify's
// previews. The ducks hear the box, not the player: their music is the box being down, at the box's volume. Spotify's
// player has no volume control, so turning the box off pauses it.
import { TRACKS } from "./tracks.js";

const API = "https://open.spotify.com/embed/iframe-api/v1";
const END_MS = 1500;  // this near the end of a song, a pause means it finished

export class SpotifyBox {
  constructor(parent, tracks = TRACKS) {
    this.order = [...tracks];
    for (let i = this.order.length - 1; i > 0; i--) {  // a new order every visit
      const j = Math.floor(Math.random() * (i + 1));
      [this.order[i], this.order[j]] = [this.order[j], this.order[i]];
    }
    this.at = 0;
    this.controller = null;
    this.ready = false;
    this.failed = !tracks.length;
    this.down = false;
    this.volume = -1;
    this.started = false;
    this.nearEnd = false;
    this.panel = document.createElement("div");
    this.panel.className = "spotify";
    this.panel.hidden = true;
    const slot = document.createElement("div");
    this.panel.appendChild(slot);
    parent.appendChild(this.panel);
    if (!this.failed) this._load(slot);
  }

  _load(slot) {
    const before = window.onSpotifyIframeApiReady;
    window.onSpotifyIframeApiReady = api => {
      if (before) before(api);
      api.createController(slot, { uri: this.order[0], width: 352, height: 152 }, controller => {
        this.controller = controller;
        controller.addListener("ready", () => { this.ready = true; this._apply(); });
        controller.addListener("playback_update", e => this._progress(e.data));
      });
    };
    const tag = document.createElement("script");
    tag.src = API;
    tag.async = true;
    tag.onerror = () => { this.failed = true; };
    document.head.appendChild(tag);
    setTimeout(() => { if (!this.ready) this.failed = true; }, 15000);
  }

  // A song that ends pauses itself at its end: then the next one plays.
  _progress({ isPaused, position, duration }) {
    if (!duration) return;
    if (isPaused && this.nearEnd && this.down && this.volume > 0) {
      this.nearEnd = false;
      this.at = (this.at + 1) % this.order.length;
      this.controller.loadUri(this.order[this.at]);
      this.controller.play();
      return;
    }
    this.nearEnd = !isPaused && position >= duration - END_MS;
  }

  // Each frame: whether the box is in the garden, and its volume 0 to 1.
  update(down, volume) {
    const changed = down !== this.down || (volume > 0) !== (this.volume > 0);
    this.down = down;
    this.volume = volume;
    this.panel.hidden = !down;
    if (changed) this._apply();
  }

  _apply() {
    if (!this.ready) return;
    if (this.down && this.volume > 0) {
      if (this.started) this.controller.resume(); else { this.controller.play(); this.started = true; }
    } else this.controller.pause();
  }
}
