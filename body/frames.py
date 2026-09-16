"""Sensory frame: one fixed-layout little-endian record per duck per body step, sent over UDP (PLAN.md Gate 3).

t is the body's monotonic clock in seconds (simulated time on the stub). Retina samples join in Gate 6.
"""
import socket

import numpy as np

FRAME = np.dtype([
    ("t", "<f8"), ("duck", "<i4"),
    ("x", "<f4"), ("y", "<f4"), ("heading", "<f4"),
    ("odor_left", "<f4"), ("odor_right", "<f4"),
    ("sugar", "<f4"), ("touch", "<f4"), ("temperature", "<f4"),
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


def receiver(duck: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((HOST, FRAME_PORT + duck))
    return s
