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
    # A ball, to a duck's eyes: how much of each eye's view it fills (nearer is more, behind is none), whether
    # it is at the duck's feet to be kicked, and 1 on the step the duck kicked it.
    ("ball_left", "<f4"), ("ball_right", "<f4"), ("ball_near", "<f4"), ("kicked", "<f4"),
    ("show", "<f4"),  # 1 on the step a duck within earshot began to sing or dance
    ("hat_near", "<f4"),  # 1 while a hat lies on the ground within this duck's reach
    # The wind on the antennae: how hard it blows, 0 to 1, and where it comes from, in radians off the
    # nose and positive to the left. A still garden sends zeros.
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


def free_port_base(wanted: int, count: int, tries: int = 40) -> int:
    """A block of `count` UDP ports nobody else holds, starting at or after `wanted`.

    Gates bind a port per duck and two running at once used to collide on the default, which kills one
    of them partway through a long run. Asking the operating system is cheaper than remembering. A garden
    someone is watching asks too: a fixed 7700 put a watched garden and a gate on the same ports (2026-09-21).
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
