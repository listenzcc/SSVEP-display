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
from mne.preprocessing import compute_proj_ecg, compute_proj_eog
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


# %% ---- 2026-04-16 ------------------------
# Function and class


# %% ---- 2026-04-16 ------------------------
# Play ground

# %%
ch_names = ['O1', 'O2', 'Oz']
freqs = np.arange(20, 41, 1)  # 20-40 Hz，步长 1 Hz

for path, raw in zip(files, raws):
    raw = raw.copy()
    raw.load_data()

    name = path.name
    print(raw)
    print(path)

    events, event_id = mne.events_from_annotations(raw)
    print(events, event_id)

    events = events[1:-1]
    events[:, -1] = 1
    event_id = {'1': 1}
    print(events, event_id)

    epochs = mne.Epochs(raw, events, event_id, tmin=-0.5, tmax=3, detrend=1)
    epochs.load_data()
    epochs.filter(l_freq=1, h_freq=50)  # 低通滤波 50Hz
    print(epochs)
    evoked = epochs[0].average()
    fig = evoked.plot_joint(show=False)
    fig.savefig(f'{name}-all-channels.png')

    epochs.pick(ch_names)
    evoked = epochs[0].average()
    fig = evoked.plot_joint(show=False)
    fig.savefig(f'{name}-selected-channels.png')

    # 使用 Morlet 小波进行时频分析
    n_cycles = freqs / 2.  # 周期数随频率增加

    power = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        return_itc=False,
        average=True,
        # decim=2  # 降采样以节省内存（可选）
    )

    # power.data shape is (68, 31, 1167), (n_channels, n_freqs, n_times)
    extent = [epochs.times[0], epochs.times[-1], freqs[0], freqs[-1]]
    fig, axes = plt.subplots(len(ch_names), 1, figsize=(12, 12))
    freq = int(name[:2])
    for ch, d, ax in zip(ch_names, power.data, axes):
        m = np.log(d)
        ax.imshow(m, aspect='auto', origin='lower', extent=extent)
        ax.axhline(freq, color='#ff000033')
        ax.set_title(f'{name}-{ch}')
    fig.tight_layout()
    fig.savefig(f'{name}.png')
    plt.show()

# %% ---- 2026-04-16 ------------------------
# Pending


# %% ---- 2026-04-16 ------------------------
# Pending
