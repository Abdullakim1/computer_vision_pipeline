import cv2
import threading
import queue
import time

class Streamer:
    """Handles threaded video capturing from various sources using OpenCV."""

    def __init__(self, source, queue_size=128):
        """Initializes the threaded video stream.

        Args:
            source (str or int): The source of the video stream.
            queue_size (int): The maximum number of frames to store in the buffer.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Successfully opened {self.source} [{self.width}x{self.height} @ {self.fps:.2f} FPS]")

        # Frame buffer
        self.Q = queue.Queue(maxsize=queue_size)
        self.stopped = False

        # Start the frame reading thread
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _update(self):
        """Private method to continuously read frames from the source."""
        while not self.stopped:
            if not self.Q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                self.Q.put(frame)
            else:
                # If the queue is full, wait a moment to avoid busy-waiting
                time.sleep(0.01)

    def get_frame(self):
        """Reads the next frame from the buffer.

        Returns:
            numpy.ndarray: The latest frame from the buffer.
        """
        return self.Q.get()

    def stop(self):
        """Signals the thread to stop."""
        self.stopped = True

    def release(self):
        """Stops the thread and releases the video capture object."""
        self.stop()
        self.thread.join() # Wait for the thread to finish
        self.cap.release()
        print("Video source released.")

    def __iter__(self):
        return self

    def __next__(self):
        # In a threaded stream, we check if the thread is alive and the queue has frames.
        # The concept of 'end of stream' is handled differently.
        if not self.thread.is_alive() and self.Q.empty():
            raise StopIteration
        return self.get_frame()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
