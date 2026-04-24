"""
File: check-using-video.py
Author: Chuncheng Zhang
Date: 2026-04-13
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check frequencies with video in 120 fps.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-13 ------------------------
# Requirements and constants
import plotly.express as px
import matplotlib.pyplot as plt
import cv2
import numpy as np

# %% ---- 2026-04-13 ------------------------
# Function and class


# %% ---- 2026-04-13 ------------------------
# Play ground
cap = cv2.VideoCapture('./SL_MO_VID_20260413_201339.mp4')

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"Total frames: {frame_count}")
print(f"Resolution: {width}x{height}")

# Pre-allocate 3D matrix (frames, height, width, channels)
# For grayscale: (frame_count, height, width)
# For color: (frame_count, height, width, 3)
video_matrix = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
video_matrix = []

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Store frame in the 3D matrix
    # Downsample by a factor of 2 for memory efficiency
    # Convert to grayscale and downsample
    video_matrix.append(np.mean(frame[::2, ::2], axis=2))
    frame_idx += 1

    # Optional: Show progress
    if frame_idx % 100 == 0:
        print(f"Read {frame_idx}/{frame_count} frames")

cap.release()
video_matrix = np.array(video_matrix)[-120*5:]

print(f"Successfully read {frame_idx} frames")
print(f"Video matrix shape: {video_matrix.shape}")

# %%
xys = [(100, 400)]

for x in [160, 360, 500, 700]:
    for y in [50, 120, 200]:
        xys.append((x, y))

plt.imshow(video_matrix[0], cmap='gray')
for x, y in xys:
    plt.plot(x, y, 'o')
plt.show()

# %%
fs = 120  # Hz

for x, y in xys:
    s = video_matrix[:, y, x]
    n = len(s)
    yf = np.fft.fft(s)
    xf = np.fft.fftfreq(n, 1/fs)
    plt.plot(xf[1:n//2], np.abs(yf[1:n//2]), linewidth=3)

for x in [20, 23, 26, 29, 31, 33, 35, 37, 32, 34, 36, 38]:
    plt.axvline(x, color='gray')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()


# %% ---- 2026-04-13 ------------------------
# Pending
x, y = 100, 400
t = np.linspace(0, 1/fs*(n-1), n)
s = video_matrix[:, y, x]
fig = px.line(x=t, y=s)
fig.show()


# %% ---- 2026-04-13 ------------------------
# Pending
