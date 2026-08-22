"""Threaded video ingestion.

Decouples frame capture (I/O bound) from downstream processing to eliminate
per-frame blocking. Works for webcams, file paths, and RTSP/HTTP streams.
"""

from __future__ import annotations

import queue
import threading
import time

import cv2


class Streamer:
    """A threaded video source that buffers frames in a bounded queue.

    Parameters
    ----------
    source : str | int
        Video file path, camera index, or stream URL understood by OpenCV.
    queue_size : int
        Maximum number of buffered frames. Bounding the queue provides
        implicit back-pressure so the capture thread blocks when the
        consumer is slower than the source.
    """

    def __init__(self, source, queue_size=128):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if not self.fps or self.fps <= 0:
            self.fps = 24.0
        print(f"[streamer] opened {source!r} {self.width}x{self.height} @ {self.fps:.2f} fps")

        self.queue = queue.Queue(maxsize=queue_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        while not self._stopped:
            if self.queue.full():
                time.sleep(0.002)
                continue
            ok, frame = self.cap.read()
            if not ok:
                self._stopped = True
                return
            self.queue.put(frame)

    # ── iterator protocol -------------------------------------------------
    def get_frame(self):
        return self.queue.get()

    def clear(self):
        self.queue.queue.clear()

    def stop(self):
        self._stopped = True

    def release(self):
        self.stop()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.cap.release()

    # context manager + iterator sugar
    def __iter__(self):
        return self

    def __next__(self):
        if self._stopped and self.queue.empty():
            raise StopIteration
        return self.get_frame()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    @property
    def stopped(self):
        return self._stopped