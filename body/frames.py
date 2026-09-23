"""Sensory frame: one fixed-layout little-endian record per duck per body step, sent over UDP (Gate 3).

t is the body's monotonic clock in seconds (simulated time on the stub). Directional senses come as a
_left/_right pair sampled at each antenna (touch: which side the other duck is on). `lum` is the
721-column hex retina, one image per eye.
"""
import socket

import numpy as np

MAX_DUCKS = 8
FRAME = np.dtype([
    ("t", "<f8"), ("duck", "<i4"),
    ("x", "<f4"), ("y", "<f4"), ("heading", "<f4"),
    ("odor_left", "<f4"), ("odor_right", "<f4"),
    ("danger_left", "<f4"), ("danger_right", "<f4"),
    ("humidity_left", "<f4"), ("humidity_right", "<f4"),
    ("temp_left", "<f4"), ("temp_right", "<f4"),
    ("touch_left", "<f4"), ("touch_right", "<f4"),
    ("duck_left", "<f4"), ("duck_right", "<f4"),  # how strongly the other ducks smell, per antenna
    ("sugar", "<f4"), ("water", "<f4"),  # water: beak can reach the pond (shore band)
    ("bumped", "<f4"),  # 1 on the step another duck headbutted this one
    ("petted", "<f4"),  # 1 on the step the player petted this one
    ("scared", "<f4"),  # 1 on the step something startled this one
    ("light", "<f4"),  # how light the garden is, 0 at night and 1 in the day
    ("music_left", "<f4"), ("music_right", "<f4"),  # how loud the music is at each ear
    ("hat", "<f4"),  # 1 while this duck is wearing a hat
    # ball: how much of each eye's view it fills (0 when behind), whether it is at the feet, 1 on the step it was kicked
    ("ball_left", "<f4"), ("ball_right", "<f4"), ("ball_near", "<f4"), ("kicked", "<f4"),
    ("drum_left", "<f4"), ("drum_right", "<f4"), ("drum_near", "<f4"), ("drummed", "<f4"),  # the same for a drum
    # the same for the pond and the tree's shade, from anywhere in the garden
    ("pond_left", "<f4"), ("pond_right", "<f4"), ("shade_left", "<f4"), ("shade_right", "<f4"),
    # Other ducks (brain/social.py). An id is a duck's number in this garden, or -1 for nobody. `near_*` is the
    # nearest duck within NEAR_M and its side; a cry is a miserable duck close by; the hand is the player's.
    # scent_*: how strongly each antenna smells each other duck, by number (own slot is 0).
    ("scent_left", "<f4", (MAX_DUCKS,)), ("scent_right", "<f4", (MAX_DUCKS,)),
    ("near_id", "<f4"), ("near_left", "<f4"), ("near_right", "<f4"),
    ("bumped_by", "<f4"), ("saw_shove_by", "<f4"), ("saw_shove_of", "<f4"), ("saw_fall_by", "<f4"),
    ("heard_alarm", "<f4"), ("heard_joy", "<f4"), ("show_by", "<f4"), ("hat_taken_by", "<f4"),
    ("cry_left", "<f4"), ("cry_right", "<f4"), ("comforted_by", "<f4"),
    ("hand_left", "<f4"), ("hand_right", "<f4"), ("hand_fed", "<f4"), ("ate_kind", "<f4"),
    ("held", "<f4"), ("thrown", "<f4"),  # 1 while the hand carries this duck; 1 on the step it was thrown
    ("show", "<f4"),  # 1 on the step a duck within earshot began to sing or dance
    ("hat_near", "<f4"),  # 1 while a hat lies on the ground within this duck's reach
    # wind strength 0 to 1, and where it comes from in radians off the nose, positive left
    ("wind", "<f4"), ("wind_from", "<f4"),
    ("ate", "<f4"),  # 1 on the step this duck took a bite
    ("drank", "<f4"),  # 1 on the step this duck took a sip
    ("swimming", "<f4"),  # 1 while the duck is in the pond past the shore band
    ("lum", "<f4", (2, 721)),  # hex-lattice retina, left eye then right (body/stub2d/retina.py)
])
MAX_BYTES = 2 * FRAME.itemsize  # the retina makes a frame ~5.9 kB; still one datagram
FRAME_PORT = 7601  # duck n sends to FRAME_PORT + n, like duck-sim's 7801 + n
HOST = "127.0.0.1"


def blank() -> np.void:
    """A frame for a duck that has not reported yet. Grey retina, not black: black is the largest
    transient the visual system can be given."""
    rec = np.zeros((), FRAME)
    rec["lum"] = 0.5  # retina.BACKGROUND, not imported to keep frames free of world imports
    for name in IDS:
        rec[name] = -1
    return rec


IDS = ("near_id", "bumped_by", "saw_shove_by", "saw_shove_of", "saw_fall_by", "show_by", "hat_taken_by", "comforted_by",
       "ate_kind")


def pack(**fields) -> bytes:
    rec = np.zeros((), FRAME)
    for name in IDS:
        rec[name] = -1  # nobody, and nothing eaten: 0 is a duck, and an orange
    for k, v in fields.items():
        rec[k] = v
    return rec.tobytes()


def unpack(data: bytes) -> np.void:
    return np.frombuffer(data, FRAME)[0]


def free_port_base(wanted: int, count: int, tries: int = 40) -> int:
    """A block of `count` UDP ports nobody else holds, starting at or after `wanted`.

    Two gates, or a gate and a watched garden, would otherwise collide on the default ports.
    """
    for attempt in range(tries):
        base = wanted + attempt * 64
        probes = []
        try:
            for i in range(count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind((HOST, base + i))
                probes.append(sock)
            return base
        except OSError:
            continue
        finally:
            for sock in probes:
                sock.close()
    raise OSError(f"no free block of {count} ports from {wanted}")


def receiver(port: int) -> socket.socket:
    """Non-blocking UDP socket bound to one duck's frame port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((HOST, port))
    s.setblocking(False)
    return s


# Fields true for one step only. They are gathered over every frame drained and wiped from a repeated frame,
# so an event counts once whether the brain runs slower or faster than the body.
ONCE = ("bumped", "petted", "scared", "ate", "drank", "kicked", "show", "hand_fed", "heard_alarm", "heard_joy", "thrown", "drummed")
ONCE_IDS = tuple(name for name in IDS if name != "near_id")


def latest(sock: socket.socket, last=None):
    """Drain the socket and return the newest frame, or last if nothing arrived, with the one-step fields
    (ONCE, ONCE_IDS) gathered over all that was drained and cleared if nothing was."""
    newest, seen = None, []
    while True:
        try:
            newest = unpack(sock.recv(MAX_BYTES))
            seen.append(newest)
        except BlockingIOError:
            break
    if newest is None:
        if last is None:
            return None
        newest = last.copy()
        for name in ONCE:
            newest[name] = 0
        for name in ONCE_IDS:
            newest[name] = -1
        return newest
    if len(seen) > 1:
        newest = newest.copy()
        for name in ONCE:
            newest[name] = max(f[name] for f in seen)
        for name in ONCE_IDS:
            newest[name] = next((f[name] for f in reversed(seen) if f[name] >= 0), -1)
    return newest


def newer(sock: socket.socket, last=None, timeout: float = 2.0):
    """Lockstep read: the newest frame later than last, waiting for it. Loopback UDP is not instant."""
    f = latest(sock)
    if f is None or (last is not None and f["t"] <= last["t"]):
        sock.settimeout(timeout)
        try:
            f = unpack(sock.recv(MAX_BYTES))
            while last is not None and f["t"] <= last["t"]:
                f = unpack(sock.recv(MAX_BYTES))
        finally:
            sock.setblocking(False)
    rest = latest(sock)  # anything that arrived behind it
    return f if rest is None else rest
