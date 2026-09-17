"""microduck's JSON-RPC 2.0 over NDJSON on Unix sockets (PLAN.md Gate 3).

Method and parameter names and units match robotd (duck-ipc-proto): robot.move {vx, vy m/s; vyaw
rad/s, positive left}, robot.head {neck_pitch, head_pitch, head_yaw, head_roll rad}, robot.do {skill},
robot.sound {tag, hold}, robot.stop, robot.init, robot.relax. Unknown members are refused, like robotd's deny_unknown_fields.
Continuous intents go as notifications; any message with an id is answered, which is how a lockstep
client knows its intent landed before it steps the stub.
"""
import json
import os
import socket
import socketserver
import threading

PARSE_ERROR, INVALID_REQUEST, METHOD_NOT_FOUND, INVALID_PARAMS = -32700, -32600, -32601, -32602

ROBOT_PARAMS = {
    "robot.move": {"vx": 0.0, "vy": 0.0, "vyaw": 0.0},
    "robot.head": {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0},
    "robot.do": {"skill": ""},
    "robot.sound": {"tag": "", "hold": None},
    "robot.stop": {},
    "robot.init": {},
    "robot.relax": {},
}


def _reply(mid, result=None, code=None, message=""):
    if code is None:
        return {"jsonrpc": "2.0", "id": mid, "result": result}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def dispatch(line: bytes, params_table: dict, call, lock: threading.Lock) -> dict | None:
    """Handle one NDJSON line. call(method, params) runs under lock; ValueError means invalid params."""
    try:
        msg = json.loads(line)
    except ValueError:
        return _reply(None, code=PARSE_ERROR, message="parse error")
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
        return _reply(msg.get("id") if isinstance(msg, dict) else None, code=INVALID_REQUEST, message="invalid request")
    mid, method, params = msg.get("id"), msg["method"], msg.get("params") or {}
    if method not in params_table:
        err = _reply(mid, code=METHOD_NOT_FOUND, message=method)
    elif not isinstance(params, dict) or set(params) - set(params_table[method]):
        err = _reply(mid, code=INVALID_PARAMS, message=f"unknown params for {method}")
    else:
        try:
            with lock:
                result = call(method, {**params_table[method], **params})
            err = None
        except (ValueError, TypeError) as e:
            err = _reply(mid, code=INVALID_PARAMS, message=str(e))
    if mid is None:
        return None
    return err or _reply(mid, result)


def serve(path: str, params_table: dict, call, lock: threading.Lock) -> socketserver.UnixStreamServer:
    """Start a threaded NDJSON server on a Unix socket in a daemon thread."""
    if os.path.exists(path):
        os.unlink(path)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            for line in self.rfile:
                reply = dispatch(line, params_table, call, lock)
                if reply is not None:
                    self.wfile.write(json.dumps(reply).encode() + b"\n")

    server = socketserver.ThreadingUnixStreamServer(path, Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class Client:
    def __init__(self, path: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.f = self.sock.makefile("rwb")
        self.next_id = 0

    def notify(self, method: str, **params) -> None:
        self.f.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}).encode() + b"\n")
        self.f.flush()

    def call(self, method: str, **params):
        self.next_id += 1
        self.f.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method, "params": params}).encode() + b"\n")
        self.f.flush()
        reply = json.loads(self.f.readline())
        if "error" in reply:
            raise RuntimeError(f"{method}: {reply['error']}")
        return reply["result"]

    def close(self) -> None:
        self.f.close()
        self.sock.close()
