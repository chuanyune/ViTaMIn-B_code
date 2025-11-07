import collections
import time

import cv2
    

class FPSHandler:
    """
    Class that handles all FPS-related operations.

    Mostly used to calculate different streams FPS, but can also be
    used to feed the video file based on its FPS property, not app performance (this prevents the video from being sent
    to quickly if we finish processing a frame earlier than the next video frame should be consumed)
    """

    _fps_bg_color = (0, 0, 0)
    _fps_color = (255, 255, 255)
    _fps_type = cv2.FONT_HERSHEY_SIMPLEX
    _fps_line_type = cv2.LINE_AA

    def __init__(self, cap=None, max_ticks=100):
        """
        Constructor that initializes the class with a video file object and a maximum ticks amount for FPS calculation

        Args:
            cap (cv2.VideoCapture, Optional): handler to the video file object
            max_ticks (int, Optional): maximum ticks amount for FPS calculation
        """
        self._timestamp = None
        self._start = None
        self._framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self._useCamera = cap is None

        self._iterCnt = 0
        self._ticks = {}

        if max_ticks < 2:  # noqa: PLR2004
            msg = f"Provided max_ticks value must be 2 or higher (supplied: {max_ticks})"
            raise ValueError(msg)

        self._maxTicks = max_ticks

    def next_iter(self):
        """Marks the next iteration of the processing loop. Will use `time.sleep` method if initialized with video file object"""
        if self._start is None:
            self._start = time.monotonic()

        if not self._useCamera and self._timestamp is not None:
            frame_delay = 1.0 / self._framerate
            delay = (self._timestamp + frame_delay) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self._timestamp = time.monotonic()
        self._iterCnt += 1

    def tick(self, name):
        """
        Marks a point in time for specified name

        Args:
            name (str): Specifies timestamp name
        """
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tick_fps(self, name):
        """
        Calculates the FPS based on specified name

        Args:
            name (str): Specifies timestamps' name

        Returns:
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if name in self._ticks and len(self._ticks[name]) > 1:
            time_diff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / time_diff if time_diff != 0 else 0.0
        return 0.0

    def fps(self):
        """
        Calculates FPS value based on `nextIter` calls, being the FPS of processing loop

        Returns:
            float: Calculated FPS or `0.0` (default in case of failure)
        """
        if self._start is None or self._timestamp is None:
            return 0.0
        time_diff = self._timestamp - self._start
        return self._iterCnt / time_diff if time_diff != 0 else 0.0

    def print_status(self):
        """Prints total FPS for all names stored in :func:`tick` calls"""
        print("=== TOTAL FPS ===")
        for name in self._ticks:
            print(f"[{name}]: {self.tick_fps(name):.1f}")

    def draw_fps(self, frame, name):
        """
        Draws FPS values on requested frame, calculated based on specified name

        Args:
            frame (numpy.ndarray): Frame object to draw values on
            name (str): Specifies timestamps' name
        """
        frame_fps = f"{name.upper()} FPS: {round(self.tick_fps(name), 1)}"
        # cv2.rectangle(frame, (0, 0), (120, 35), (255, 255, 255), cv2.FILLED)
        draw_text(
            frame,
            frame_fps,
            (5, 15),
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
        )

        if "nn" in self._ticks:
            draw_text(
                frame,
                f"NN FPS:  {round(self.tick_fps('nn'), 1)}",
                (5, 30),
                color=(255, 255, 255),
                bg_color=(0, 0, 0),
            )


def draw_text(
    frame,
    text,
    org,
    color=(255, 255, 255),
    bg_color=(128, 128, 128),
    font_scale=0.5,
    thickness=1,
):
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        bg_color,
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
