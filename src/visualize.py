# src/visualize.py

import numpy as np
import tensorflow as tf
from IPython import display
from matplotlib import pyplot as plt

# Hàm vẽ waveform
def plot_wave(waveform, label):
    plt.figure(figsize=(10, 3))
    plt.title(label)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])
    plt.ylim([-1, 1])
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Hàm chuyển sang spectrogram
def get_spectrogram(waveform):
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    return spec[..., tf.newaxis]

# Hàm vẽ spectrogram
def plot_spectrogram(spectrogram, label):
    spec = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spec.T + np.finfo(float).eps)
    plt.figure(figsize=(10, 3))
    plt.title(label)
    plt.imshow(log_spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()

# Demo trên 1 sample
if __name__ == "__main__":
    from data_prep import train_ds, label_names
    # Lấy 1 batch và 1 sample đầu tiên
    audio_batch, label_batch = next(iter(train_ds))
    waveform = audio_batch[0].numpy()
    label = label_names[label_batch[0].numpy()]

    # Phát audio
    display.display(display.Audio(waveform, rate=16000))

    # Vẽ
    plot_wave(waveform, label)
    spec = get_spectrogram(waveform)
    plot_spectrogram(spec, label)
