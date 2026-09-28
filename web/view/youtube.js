// The music box's playlist: a YouTube playlist in YouTube's own embedded player, shown while the box is in the garden
// and paused when it is picked up. YouTube asks that its player stay visible while it plays, so it sits in the corner.
// A video whose owner does not allow embedding is skipped. The ducks hear the box, not the video: their music is
// the box being down, at the box's volume, as in the native garden.

export const PLAYLIST = "PL7CAA462F393D487B";
const API = "https://www.youtube.com/iframe_api";

export class YouTubeBox {
  constructor(parent, playlist = PLAYLIST) {
    this.playlist = playlist;
    this.player = null;
    this.ready = false;
    this.failed = false;
    this.title = "";
    this.down = false;
    this.volume = -1;
    this.panel = document.createElement("div");
    this.panel.className = "youtube";
    this.panel.hidden = true;
    const slot = document.createElement("div");
    this.panel.appendChild(slot);
    parent.appendChild(this.panel);
    this._load(slot);
  }

  _load(slot) {
    const make = () => {
      try {
        this.player = new YT.Player(slot, {
          width: 356, height: 200,
          playerVars: { listType: "playlist", list: this.playlist, playsinline: 1, rel: 0, modestbranding: 1 },
          events: {
            onReady: () => { this.ready = true; this.player.setShuffle(true); this._apply(); },
            onStateChange: () => { this.title = this.player.getVideoData?.().title || this.title; },
            onError: () => { this.player.nextVideo(); },  // not embeddable, or gone: the next one
          },
        });
      } catch { this.failed = true; }
    };
    if (window.YT && window.YT.Player) { make(); return; }
    const before = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => { if (before) before(); make(); };
    const tag = document.createElement("script");
    tag.src = API;
    tag.onerror = () => { this.failed = true; };
    document.head.appendChild(tag);
    setTimeout(() => { if (!this.ready) this.failed = true; }, 15000);
  }

  // Each frame: whether the box is in the garden, and its volume 0 to 1.
  update(down, volume) {
    const changed = down !== this.down || volume !== this.volume;
    this.down = down;
    this.volume = volume;
    this.panel.hidden = !down;
    if (changed) this._apply();
  }

  _apply() {
    if (!this.ready) return;
    this.player.setVolume(Math.round(this.volume * 100));
    if (this.down && this.volume > 0) {
      if (this.player.getPlayerState() !== YT.PlayerState.PLAYING) this.player.playVideo();
    } else this.player.pauseVideo();
  }
}
