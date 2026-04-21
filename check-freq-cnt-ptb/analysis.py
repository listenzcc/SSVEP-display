"""
File: analysis.py
Author: Chuncheng Zhang
Date: 2026-04-16
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis the eeg response for SSVEP blinking.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-16 ------------------------
# Requirements and constants
import plotly.express as px
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
import numpy as np
import matplotlib.pyplot as plt
import mne
import pandas as pd
from pathlib import Path
from rich import print

# %%
files = sorted(Path('./ygs').iterdir())
print(files)

raws = [mne.io.read_raw_cnt(e) for e in files]
print(raws)

raw = raws[0]
raw.load_data()
print(raw)

# %%
events, event_id = mne.events_from_annotations(raw)
print(events, event_id)

# %%
epochs = mne.Epochs(raw, events, event_id, tmin=-0.5,
                    tmax=3.5, detrend=1, preload=True)
print(epochs)

# %%
fig = plt.figure(figsize=(12, 3))
for i in range(len(epochs)):
    evoked = epochs[i].average()
    x = evoked.times
    y = evoked.data[7]
    plt.plot(x, y, label=f'{i}')
plt.legend()
plt.show()

# %%
fs = int(raw.info['sfreq'])
mask = (x > 0.1) & (x < 2.5)
y = y[mask]
n = len(y)
yf = np.fft.fft(y)
xf = np.fft.fftfreq(n, 1/fs)
xf = xf[10:n//2]
yf = yf[10:n//2]
yf = yf[xf < 50]
xf = xf[xf < 50]

plt.plot(xf, np.abs(yf), linewidth=3)
plt.show()

# %%

# %%

df = []

for i in range(len(epochs)):
    evoked = epochs[i].average()
    x = evoked.times
    y = evoked.data[7]
    df.append(pd.DataFrame({'time': x, 'amplitude': y, 'epoch': f'{i}'}))

df = pd.concat(df, ignore_index=True)
print(df.head())

fig = px.line(df, x='time', y='amplitude', color='epoch')
fig.show()


# %%

# %% ---- 2026-04-16 ------------------------
# Function and class


# %% ---- 2026-04-16 ------------------------
# Play ground

# %%

# %% ---- 2026-04-16 ------------------------
# Pending


# %% ---- 2026-04-16 ------------------------
# Pending

# %%
