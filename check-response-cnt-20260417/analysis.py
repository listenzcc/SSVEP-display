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
files = sorted(Path('./FlashCnt').iterdir())
print(files)

raws = [mne.io.read_raw_cnt(e) for e in files]
print(raws)


# %% ---- 2026-04-16 ------------------------
# Function and class


# %% ---- 2026-04-16 ------------------------
# Play ground

# %%
ch_names = ['Oz', 'POz', 'Pz']
freqs = np.arange(20, 41, 0.5)  # 20-40 Hz，步长 0.5 Hz

for path, raw in zip(files, raws):
    raw = raw.copy()
    raw.load_data()

    name = path.name
    print(raw)
    print(path)
    freq = float(name.replace('Hz.cnt', ''))

    events, event_id = mne.events_from_annotations(raw)
    print(events, event_id)

    events = events[1:-1]
    events[:, -1] = 1
    event_id = {'1': 1}
    print(events, event_id)

    epochs = mne.Epochs(raw, events, event_id, tmin=-0.5, tmax=3, detrend=0)
    if name.startswith('31'):
        epochs = epochs[1:]
    epochs.load_data()
    epochs.filter(l_freq=1, h_freq=40)  # 低通滤波 50Hz
    print(epochs)
    evoked = epochs[0].average()
    fig = evoked.plot_joint(show=False)
    fig.savefig(f'{name}-all-channels.png')

    epochs.pick(ch_names)
    evoked = epochs[0].average()
    fig = evoked.plot_joint(show=False)
    fig.savefig(f'{name}-selected-channels.png')

    # FFT on epochs data
    data = epochs.get_data()
    # Transpose data shape to (n_channels, n_epochs, n_times)
    data = data.transpose((1, 0, 2))
    print(data.shape)
    fig, axes = plt.subplots(len(ch_names), 1, figsize=(12, 12))
    fs = epochs.info['sfreq']
    for ch, yy, ax in zip(ch_names, data, axes):
        yf_stack = []
        for y in yy:
            n = len(y)
            yf = np.abs(np.fft.fft(y))
            xf = np.fft.fftfreq(n, 1/fs)
            xf = xf[10:n//2]
            yf = yf[10:n//2]
            yf = yf[xf < freqs[-1]]
            xf = xf[xf < freqs[-1]]
            ax.plot(xf, yf, linewidth=1, color='gray')
            yf_stack.append(yf)

        yf = np.mean(yf_stack, axis=0)
        ax.plot(xf, yf, linewidth=3, color='red')
        ax.axvline(freq, color='pink')
        ax.set_title(f'{name}-{ch}')

    fig.tight_layout()
    fig.savefig(f'{name}-fft.png')
    plt.show()

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
    fig, axes = plt.subplots(len(ch_names)+1, 1, figsize=(12, 12))
    ax_last = axes[-1]
    for ch, d, ax in zip(ch_names, power.data, axes):
        m = np.log(d)
        ax.imshow(m, aspect='auto', origin='lower', extent=extent)
        ax.axhline(freq, color='#ff000033')
        ax_last.plot(freqs, np.mean(m[:, epochs.times > 0], axis=1), label=ch)
        ax_last.axvline(freq, color='black')
        ax.set_title(f'{name}-{ch}')
    ax_last.legend()
    fig.tight_layout()
    fig.savefig(f'{name}.png')
    plt.show()

# %% ---- 2026-04-16 ------------------------
# Pending
raw.ch_names

# %% ---- 2026-04-16 ------------------------
# Pending
fig = mne.viz.plot_sensors(raw.info, show_names=True)
plt.show()

# %%
