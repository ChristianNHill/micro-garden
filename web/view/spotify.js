// The music box's playlist, played two ways. By default it is Spotify's embedded player, shown while the box is in
// the garden: whole songs only for a visitor whose browser lets the embed see a Spotify login, previews otherwise.
// A visitor with Spotify Premium can connect their account instead: the garden then plays the whole playlist itself
// through Spotify's Web Playback SDK, shuffled, and the box's volume is the music's volume. The ducks hear the box,
// not the player: their music is the box being down, at the box's volume, as in the native garden.

export const PLAYLIST = "2vg04PEDQePvRsS7PD3TyS";  // the playlist's id, from its Spotify link: open.spotify.com/playlist/<id>
// The Client ID of the Spotify app registered for the garden (developer.spotify.com/dashboard). Empty: no Connect button.
export const CLIENT_ID = "af3086c3d8c945d1b7e156b5d4b3aca4";
const EMBED_API = "https://open.spotify.com/embed/iframe-api/v1";
const SDK = "https://sdk.scdn.co/spotify-player.js";
const SCOPES = "streaming user-read-email user-read-private user-read-playback-state user-modify-playback-state";
const SAVED = "micro-garden-spotify";  // the connected account's tokens, in this browser only

const store = {
  get: key => { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
  set: (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch {} },
  drop: key => { try { localStorage.removeItem(key); sessionStorage.removeItem(key); } catch {} },
};
const redirect = () => location.origin + location.pathname;
const base64url = bytes => btoa(String.fromCharCode(...new Uint8Array(bytes))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

export class SpotifyBox {
  // `say` shows a line to the visitor.
  constructor(parent, say = () => {}, playlist = PLAYLIST) {
    this.playlist = playlist;
    this.say = say;
    this.controller = null;  // the embed
    this.ready = false;
    this.failed = !playlist;
    this.title = "";
    this.down = false;
    this.volume = -1;
    this.player = null;  // the Web Playback SDK, once connected
    this.device = null;
    this.connected = false;
    this.connecting = false;
    this.startedHere = false;
    this.panel = document.createElement("div");
    this.panel.className = "spotify";
    this.panel.hidden = true;
    const slot = document.createElement("div");
    this.panel.appendChild(slot);
    this.now = document.createElement("div");  // what is playing, once connected
    this.now.className = "spotify-now";
    this.now.hidden = true;
    this.now.innerHTML = '<div class="spotify-track"><b></b><span></span></div><button type="button">Disconnect</button>';
    this.now.querySelector("button").addEventListener("click", e => { e.stopPropagation(); this.disconnect(); });
    this.panel.appendChild(this.now);
    parent.appendChild(this.panel);
    this.connectButton = document.createElement("button");
    this.connectButton.className = "spotify-connect";
    this.connectButton.type = "button";
    this.connectButton.textContent = "Connect Spotify for whole songs";
    this.connectButton.hidden = true;
    this.connectButton.addEventListener("click", e => { e.stopPropagation(); this.connect(); });
    parent.appendChild(this.connectButton);
    if (playlist) this._loadEmbed(slot);
    if (CLIENT_ID) this._resume();
  }

  // Each frame: whether the box is in the garden, and its volume 0 to 1.
  update(down, volume) {
    const changed = down !== this.down || (volume > 0) !== (this.volume > 0);
    const louder = volume !== this.volume;
    this.down = down;
    this.volume = volume;
    this.panel.hidden = !down;
    this.panel.classList.toggle("connected", this.connected);
    this.connectButton.hidden = !(CLIENT_ID && down && !this.connected && !this.connecting);
    if (this.connected) {
      if (changed) this._playHere();
      else if (louder && volume > 0) this.player.setVolume(volume);
    } else if (changed) this._applyEmbed();
  }

  // Only a connected player's volume can be set from the page; the embed's cannot.
  get canSetVolume() { return this.connected; }

  // ---- the embed

  _loadEmbed(slot) {
    const before = window.onSpotifyIframeApiReady;
    window.onSpotifyIframeApiReady = api => {
      if (before) before(api);
      const narrow = matchMedia("(max-width: 760px)").matches;  // a phone gets Spotify's compact player
      api.createController(slot, { uri: `spotify:playlist:${this.playlist}`, width: narrow ? "100%" : 352, height: narrow ? 80 : 152 }, controller => {
        this.controller = controller;
        controller.addListener("ready", () => { this.ready = true; this._applyEmbed(); });
      });
    };
    const tag = document.createElement("script");
    tag.src = EMBED_API;
    tag.async = true;
    tag.onerror = () => { this.failed = true; };
    document.head.appendChild(tag);
    setTimeout(() => { if (!this.ready && !this.connected) this.failed = true; }, 15000);
  }

  _applyEmbed() {
    if (!this.ready || this.connected) return;
    if (this.down && this.volume > 0) {
      if (this.started) this.controller.resume(); else { this.controller.play(); this.started = true; }
    } else this.controller.pause();
  }

  // ---- connecting an account (authorization code with PKCE: no server and no secret)

  async connect() {
    const verifier = base64url(crypto.getRandomValues(new Uint8Array(64)));
    const challenge = base64url(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(verifier)));
    const state = base64url(crypto.getRandomValues(new Uint8Array(16)));
    try { sessionStorage.setItem(SAVED, JSON.stringify({ verifier, state })); } catch {}
    const q = new URLSearchParams({ client_id: CLIENT_ID, response_type: "code", redirect_uri: redirect(), scope: SCOPES,
                                    code_challenge_method: "S256", code_challenge: challenge, state });
    location.assign(`https://accounts.spotify.com/authorize?${q}`);
  }

  disconnect() {
    store.drop(SAVED);
    if (this.player) this.player.disconnect();
    this.player = null; this.device = null; this.connected = false; this.startedHere = false;
    this.now.hidden = true;
    this.say("Spotify disconnected. The box plays Spotify's own player again");
    this._applyEmbed();
  }

  // Back from Spotify with a code, or a saved account from before: get a token and start the player.
  async _resume() {
    const params = new URLSearchParams(location.search);
    let pending = null;
    try { pending = JSON.parse(sessionStorage.getItem(SAVED)); } catch {}
    if (params.has("code") || params.has("error")) {
      history.replaceState(null, "", redirect());  // the code works once; keep it out of the address bar
      try { sessionStorage.removeItem(SAVED); } catch {}
      if (params.has("error") || !pending || params.get("state") !== pending.state) {
        this.say("Spotify was not connected");
        return;
      }
      this.connecting = true;
      const got = await this._token({ grant_type: "authorization_code", code: params.get("code"), redirect_uri: redirect(), code_verifier: pending.verifier });
      this.connecting = false;
      if (!got) { this.say("Spotify would not connect. Try again later"); return; }
    }
    if (store.get(SAVED)) this._startSdk();
  }

  async _token(body) {
    try {
      const r = await fetch("https://accounts.spotify.com/api/token", {
        method: "POST", headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ client_id: CLIENT_ID, ...body }),
      });
      if (!r.ok) return null;
      const t = await r.json();
      const saved = store.get(SAVED) || {};
      store.set(SAVED, { access: t.access_token, refresh: t.refresh_token || saved.refresh, until: Date.now() + (t.expires_in - 60) * 1000 });
      return t.access_token;
    } catch { return null; }
  }

  async _access() {
    const saved = store.get(SAVED);
    if (!saved) return null;
    if (Date.now() < saved.until) return saved.access;
    return this._token({ grant_type: "refresh_token", refresh_token: saved.refresh });
  }

  _startSdk() {
    window.onSpotifyWebPlaybackSDKReady = () => {
      const player = new window.Spotify.Player({ name: "Micro Garden", volume: Math.max(this.volume, 0.5),
                                                 getOAuthToken: done => this._access().then(t => t && done(t)) });
      player.addListener("ready", ({ device_id }) => {
        this.player = player; this.device = device_id; this.connected = true;
        this.say("Spotify connected. The music box plays whole songs");
        this._playHere();
      });
      player.addListener("not_ready", () => { this.device = null; });
      player.addListener("account_error", () => this._giveUp("Whole songs need Spotify Premium. The box plays previews instead"));
      player.addListener("initialization_error", () => this._giveUp("This browser cannot play Spotify here. The box plays previews instead"));
      player.addListener("authentication_error", () => { store.drop(SAVED); this._giveUp("Spotify asked to connect again"); });
      player.addListener("player_state_changed", state => this._showTrack(state));
      player.connect();
    };
    const tag = document.createElement("script");
    tag.src = SDK;
    tag.async = true;
    document.head.appendChild(tag);
    // a tap or click lets the browser start the sound, which phones and Safari insist on
    addEventListener("pointerdown", () => this.player?.activateElement?.(), { once: true });
  }

  _giveUp(line) {
    if (this.player) this.player.disconnect();
    this.player = null; this.connected = false;
    this.now.hidden = true;
    this.say(line);
    this._applyEmbed();
  }

  // The box is down and on: play the playlist on this page's player, shuffled, from a random song. Otherwise pause.
  async _playHere() {
    if (!this.player || !this.device) return;
    this.now.hidden = false;
    if (this.controller) this.controller.pause();
    if (!(this.down && this.volume > 0)) { this.player.pause(); return; }
    if (this.startedHere) { this.player.setVolume(this.volume); this.player.resume(); return; }
    this.startedHere = true;
    const token = await this._access();
    const api = (path, init = {}) => fetch(`https://api.spotify.com/v1${path}`,
      { ...init, headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" } });
    let songs = 1;
    try { songs = (await (await api(`/playlists/${this.playlist}?fields=tracks.total`)).json()).tracks.total || 1; } catch {}
    await api(`/me/player/play?device_id=${this.device}`, { method: "PUT",
      body: JSON.stringify({ context_uri: `spotify:playlist:${this.playlist}`, offset: { position: Math.floor(Math.random() * songs) } }) });
    await api(`/me/player/shuffle?state=true&device_id=${this.device}`, { method: "PUT" });
    this.player.setVolume(this.volume);
  }

  _showTrack(state) {
    const track = state?.track_window?.current_track;
    if (!track) return;
    this.now.querySelector("b").textContent = track.name;
    this.now.querySelector("span").textContent = track.artists.map(a => a.name).join(", ");
  }
}
