"""A simulated microduck's head camera, as the fly's two eyes see it (Gate 11).

`duck-body --cameras a` serves each camera on its own TCP port, 7901 + n: four bytes of little-endian
length, then a 640x360 UYVY frame, at 15 fps. The camera is mounted a quarter turn off, as on the real
robot, so each frame is turned upright first. Reading it needs no GStreamer, but `scripts/duck-sim` refuses
to finish without GStreamer when cameras are on; the body server is already up then, so send `robot.enable`
yourself.

Upright, the camera sees 45 degrees across (MuJoCo's default fovy) and 73 up and down. That falls inside the
frontal 40-degree overlap of the fly's two eyes, so both eyes see it. Columns outside the camera get the
frame's mean, not a fixed grey, so the camera's edge is not a standing contrast edge for the motion detectors.
"""
import socket
import struct
import threading

import numpy as np

from body.stub2d.retina import EYE_AZ, HEX_AZ, HEX_EL, N_HEX

FRAME_PORT = 7901
WIDTH, HEIGHT = 640, 360  # as sent; upright it is 360 across and 640 tall
FOV_ACROSS = np.radians(45.0)  # MuJoCo's default fovy, which lies across the upright frame


def upright_luma(frame: bytes) -> np.ndarray:
    """(640, 360) brightness in 0-1 from one UYVY frame, turned the right way up."""
    y = np.frombuffer(frame, np.uint8).reshape(HEIGHT, WIDTH, 2)[:, :, 1]
    return np.rot90(y, k=-1).astype(np.float32) / 255.0


def _pixels():
    """Which upright pixel each hex column of each eye looks through, and which columns the camera sees."""
    az = EYE_AZ[:, None] - HEX_AZ[None, :]  # left of straight ahead, per eye and column
    el = np.broadcast_to(HEX_EL, az.shape)
    f = (HEIGHT / 2) / np.tan(FOV_ACROSS / 2)  # pixels per unit tangent; 360 is the upright width
    u = HEIGHT / 2 - f * np.tan(az)  # image x runs to the right, azimuth to the left
    v = WIDTH / 2 - f * np.tan(el) / np.cos(az)
    seen = (np.abs(az) < np.pi / 2) & (u >= 0) & (u < HEIGHT) & (v >= 0) & (v < WIDTH)
    return np.clip(v, 0, WIDTH - 1).astype(int), np.clip(u, 0, HEIGHT - 1).astype(int), seen


_ROW, _COL, SEEN = _pixels()


def to_retina(luma: np.ndarray) -> np.ndarray:
    """(2, 721) hex-lattice brightness for the left and right eye from one upright frame."""
    return np.where(SEEN, luma[_ROW, _COL], luma.mean()).astype(np.float32)


class SimCamera:
    """The newest frame from one duck's camera, read on a thread so the garden never waits for one."""

    def __init__(self, port: int = FRAME_PORT):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.luma = None
        self.frames = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        try:
            while True:
                n = struct.unpack("<I", self.sock.recv(4, socket.MSG_WAITALL))[0]
                self.luma = upright_luma(self.sock.recv(n, socket.MSG_WAITALL))
                self.frames += 1
        except (OSError, struct.error):
            pass  # the simulator went away; keep the last frame

    def retina(self) -> np.ndarray | None:
        return None if self.luma is None else to_retina(self.luma)

    def close(self) -> None:
        self.sock.close()


def demo() -> None:
    """The geometry, on a frame made here: bright sky over dark ground, and a bright post to the left."""
    luma = np.full((WIDTH, HEIGHT), 0.2, np.float32)
    luma[: WIDTH // 2] = 0.8  # sky above the horizon
    luma[:, : HEIGHT // 4] = 1.0  # a post at the left edge of the view
    r = to_retina(luma)
    assert SEEN.sum(1).min() > 20, f"each eye should see the camera through some columns: {SEEN.sum(1)}"
    for eye in (0, 1):
        s = SEEN[eye]
        up, down = s & (HEX_EL > 0.15), s & (HEX_EL < -0.15)
        assert r[eye, up & (EYE_AZ[eye] - HEX_AZ < 0.1)].mean() > r[eye, down & (EYE_AZ[eye] - HEX_AZ < 0.1)].mean() + 0.3, "sky over ground"
        leftmost = s & (EYE_AZ[eye] - HEX_AZ > np.radians(15))
        assert leftmost.any() and r[eye, leftmost].mean() > 0.9, "the post is to the left in both eyes"
    assert np.allclose(r[~SEEN], luma.mean()), "columns outside the camera get the frame's mean"
    print(f"ok  camera seen through {SEEN.sum(1).tolist()} of {N_HEX} columns a eye; sky over ground; left is left")


if __name__ == "__main__":
    demo()
