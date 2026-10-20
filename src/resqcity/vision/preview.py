"""Minimal MJPEG preview server for anonymised, annotated frames (dashboard camera tiles).

GET / or /stream.mjpg -> multipart/x-mixed-replace stream of the latest frame.
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BOUNDARY = "resqcityframe"
JPEG_QUALITY = 80
FRAME_INTERVAL_S = 0.1


class FrameStore:
    def __init__(self) -> None:
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()

    def update(self, image: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def latest(self) -> bytes | None:
        with self._lock:
            return self._jpeg


def serve(store: FrameStore, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path not in ("/", "/stream.mjpg"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    jpeg = store.latest()
                    if jpeg is not None:
                        head = (
                            f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n"
                        )
                        self.wfile.write(head.encode())
                        self.wfile.write(jpeg + b"\r\n")
                    time.sleep(FRAME_INTERVAL_S)
            except (BrokenPipeError, ConnectionResetError):
                return

        def log_message(self, *args: object) -> None:  # silence per-request logging
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
