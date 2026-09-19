"""Sensory frame: one fixed-layout little-endian record per duck per body step, sent over UDP (PLAN.md Gate 3).

t is the body's monotonic clock in seconds (simulated time on the stub). Directional senses come as a
_left/_right pair sampled at each antenna (touch: which side the other duck is on). `lum` is the
721-column hex retina, one image per eye.
"""
import socket

import numpy as np

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
    ("petted", "<f4"),  # 1 on the step the player petted this one (PLAN.md Gate 8)
    ("scared", "<f4"),  # 1 on the step something startled this one
    ("light", "<f4"),  # how light the garden is, 0 at night and 1 in the day
    ("music_left", "<f4"), ("music_right", "<f4"),  # how loud the music is at each ear (Gate 8b)
    ("hat", "<f4"),  # 1 while this duck is wearing a hat
    ("ate", "<f4"),  # 1 on the step this duck took a bite
    ("drank", "<f4"),  # 1 on the step this duck took a sip
    ("swimming", "<f4"),  # 1 while the duck is in the pond past the shore band
    ("lum", "<f4", (2, 721)),  # hex-lattice retina, left eye then right (body/stub2d/retina.py)
])
MAX_BYTES = 2 * FRAME.itemsize  # the retina makes a frame ~5.9 kB; still one datagram
FRAME_PORT = 7601  # duck n sends to FRAME_PORT + n, like duck-sim's 7801 + n
HOST = "127.0.0.1"


def blank() -> np.void:
    """A frame for a duck that has not reported yet. Grey retina, not black: an all-zero record would
    read as pitch darkness in both eyes, the largest transient the visual system can be given."""
    rec = np.zeros((), FRAME)
    rec["lum"] = 0.5  # body.stub2d.retina.BACKGROUND; named here to keep frames free of world imports
    return rec


def pack(**fields) -> bytes:
    rec = np.zeros((), FRAME)
    for k, v in fields.items():
        rec[k] = v
    return rec.tobytes()


def unpack(data: bytes) -> np.void:
    return np.frombuffer(data, FRAME)[0]


def receiver(port: int) -> socket.socket:
    """Non-blocking UDP socket bound to one duck's frame port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((HOST, port))
    s.setblocking(False)
    return s


def latest(sock: socket.socket, last=None):
    """Drain the socket and return the newest frame, or last if nothing arrived."""
    while True:
        try:
            last = unpack(sock.recv(MAX_BYTES))
        except BlockingIOError:
            return last


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
    return latest(sock, f)
