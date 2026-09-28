// The music box's playlist: a Spotify playlist in Spotify's own embedded player, shown while the box is in the garden
// and paused when it is picked up. A visitor signed in to Spotify in this browser hears whole songs; anyone else hears
// Spotify's previews. The ducks hear the box, not the player: their music is the box being down, at the box's
// volume, as in the native garden. Spotify's embed has no volume control, so the box's volume only reaches the
// ducks, and turning the box off pauses the player.

export const PLAYLIST = "";  // the playlist's id, from its Spotify link: open.spotify.com/playlist/<id>
const API = "https://open.spotify.com/embed/iframe-api/v1";

export class SpotifyBox {
  constructor(parent, playlist = PLAYLIST) {
    this.playlist = playlist;
    this.controller = null;
    this.ready = false;
    this.failed = !playlist;
    this.title = "";
    this.down = false;
    this.volume = -1;
    this.panel = document.createElement("div");
    this.panel.className = "spotify";
    this.panel.hidden = true;
    const slot = document.createElement("div");
    this.panel.appendChild(slot);
    parent.appendChild(this.panel);
    if (playlist) this._load(slot);
  }

  _load(slot) {
    const before = window.onSpotifyIframeApiReady;
    window.onSpotifyIframeApiReady = api => {
      if (before) before(api);
      api.createController(slot, { uri: `spotify:playlist:${this.playlist}`, width: 352, height: 152 }, controller => {
        this.controller = controller;
        controller.addListener("ready", () => { this.ready = true; this._apply(); });
      });
    };
    const tag = document.createElement("script");
    tag.src = API;
    tag.async = true;
    tag.onerror = () => { this.failed = true; };
    document.head.appendChild(tag);
    setTimeout(() => { if (!this.ready) this.failed = true; }, 15000);
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
