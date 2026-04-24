"""
File: fps_ruler.py
Author: Chuncheng Zhang
Date: 2025-10-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    FPS ruler.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-10-09 ------------------------
# Requirements and constants
import time
import json
from datetime import datetime
from collections import deque, defaultdict

# %% ---- 2025-10-09 ------------------------
# Function and class


class FPSRuler:
    timestamps: deque
    intervals: defaultdict

    def __init__(self, max_samples=100):
        """
        Initialize the frame rate counter.

        Args:
            max_samples (int): Maximum number of timestamps to store for calculating frame rate.
        """
        self.timestamps = deque(maxlen=max_samples)
        self.intervals = defaultdict(int)

    def update(self, interval=None):
        """
        Update the frame rate counter with the current timestamp.

        :param interval: The time interval for the current frame (in milliseconds), used for categorizing frame rates.
        """
        self.timestamps.append(time.time())
        if interval is not None:
            self.intervals[int(interval*1e3)] += 1

    def summary(self):
        print(self.intervals)
        json.dump(
            self.intervals,
            open(
                f'summary-{datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")}.json', 'w'),
            indent=4)

    def get_fps(self):
        """
        Calculate and return the current frame rate.

        Returns:
            float: The calculated frame rate, or 0.0 if not enough data is available.
        """
        if len(self.timestamps) < 2:
            return 0.0
        time_deltas = [self.timestamps[i] - self.timestamps[i - 1]
                       for i in range(1, len(self.timestamps))]
        avg_delta = sum(time_deltas) / len(time_deltas)
        rate = 1.0 / avg_delta if avg_delta > 0 else 0.0
        return rate

# %% ---- 2025-10-09 ------------------------
# Play ground


# %% ---- 2025-10-09 ------------------------
# Pending


# %% ---- 2025-10-09 ------------------------
# Pending
