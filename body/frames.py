"""Sensory frame: one fixed-layout little-endian record per duck per body step, sent over UDP (PLAN.md Gate 3).

t is the body's monotonic clock in seconds (simulated time on the stub). Directional senses come as a
_left/_right pair sampled at each antenna (touch: which side the other duck is on). Retina joins in Gate 6.
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
    ("sugar", "<f4"), ("water", "<f4"),  # water: beak can reach the pond (shore band)
    ("bumped", "<f4"),  # 1 on the step another duck headbutted this one
    ("ate", "<f4"),  # 1 on the step this duck took a bite
    ("drank", "<f4"),  # 1 on the step this duck took a sip
    ("swimming", "<f4"),  # 1 while the duck is in the pond past the shore band
])
FRAME_PORT = 7601  # duck n sends to FRAME_PORT + n, like duck-sim's 7801 + n
HOST = "127.0.0.1"


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
            last = unpack(sock.recv(1024))
        except BlockingIOError:
            return last


def newer(sock: socket.socket, last=None, timeout: float = 2.0):
    """Lockstep read: the newest frame later than last, waiting for it. Loopback UDP is not instant."""
    f = latest(sock)
    if f is None or (last is not None and f["t"] <= last["t"]):
        sock.settimeout(timeout)
        try:
            f = unpack(sock.recv(1024))
            while last is not None and f["t"] <= last["t"]:
                f = unpack(sock.recv(1024))
        finally:
            sock.setblocking(False)
    return latest(sock, f)
