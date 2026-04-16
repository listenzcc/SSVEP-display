"""
File: analysis.py
Author: Chuncheng Zhang
Date: 2026-04-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis cnt files.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-15 ------------------------
# Requirements and constants
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from pathlib import Path
from rich import print

# %%
files = sorted(Path('./eeg').iterdir())
print(files)

raws = [mne.io.read_raw_cnt(e) for e in files]
print(raws)


# %% ---- 2026-04-15 ------------------------
# Function and class


# %% ---- 2026-04-15 ------------------------
# Play ground
fs = 1000  # Hz

fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = np.ravel(axes)

for file, raw, ax in zip(files, raws, axes):
    name = file.name

    fs_real = int(name.split('_')[1][:2])

    raw = raw.copy()
    y = raw.get_data()[62][-8*fs:]

    n = len(y)
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(n, 1/fs)
    xf = xf[10:n//2]
    yf = yf[10:n//2]
    yf = yf[xf < 50]
    xf = xf[xf < 50]

    ax.plot(xf, np.abs(yf), linewidth=3)
    ax.axvline(fs_real, color='gray')
    ax.set_title(f'{file.name}')

fig.tight_layout()

plt.show()

# %% ---- 2026-04-15 ------------------------
# Pending


# %% ---- 2026-04-15 ------------------------
# Pending
