import tensorflow as tf
import numpy as np
from data_prep import squeeze, label_names
from model import get_model

# Load model
model = tf.keras.models.load_model('audio_recog_cnn.keras')

# Đọc file WAV mới
path = 'data/mini_speech_commands/yes/004ae714_nohash_0.wav'
audio_binary = tf.io.read_file(path)
waveform, _ = tf.audio.decode_wav(audio_binary,
                                  desired_channels=1,
                                  desired_samples=16000)
waveform = tf.squeeze(waveform, axis=-1)

# Spectrogram
spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
spec = tf.abs(spec)[..., tf.newaxis]
spec = tf.expand_dims(spec, 0)

# Dự đoán
pred = model.predict(spec)
probs = tf.nn.softmax(pred[0])
print("Prediction:", label_names[np.argmax(probs)])
