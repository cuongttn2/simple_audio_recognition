import tensorflow as tf

from data_prep import train_ds, val_ds, label_names
from model import get_model


# Tạo spectrogram dataset
def get_spectrogram(waveform):
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    return spec[..., tf.newaxis]


def to_spec(ds):
    return ds.map(lambda x, y: (get_spectrogram(x), y), num_parallel_calls=tf.data.AUTOTUNE)


train_set = to_spec(train_ds)
val_set_full = to_spec(val_ds)

# Chia val thành val và test
val_size = val_set_full.cardinality() // 2
val_set = val_set_full.take(val_size)
test_set = val_set_full.skip(val_size)

# Xây model
input_shape = next(iter(train_set))[0][0].shape
num_labels = len(label_names)
model = get_model(input_shape, num_labels)

# Compile & Fit
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

EPOCHS = 10
history = model.fit(train_set, validation_data=val_set, epochs=EPOCHS)

# --- In shape của các tập ---
print("Train set shape:", train_set.element_spec[0].shape)
print("Validation set shape:", val_set.element_spec[0].shape)
print("Testing set shape:", test_set.element_spec[0].shape)

# --- Vẽ Loss & Accuracy ---
import matplotlib.pyplot as plt

metrics = history.history

plt.figure(figsize=(12, 5))
# Loss
plt.subplot(1, 2, 1)
plt.plot(history.epoch, metrics['loss'], label='train loss')
plt.plot(history.epoch, metrics['val_loss'], label='val loss')
plt.xlabel('Epoch');
plt.ylabel('Loss');
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.epoch, metrics['accuracy'], label='train acc')
plt.plot(history.epoch, metrics['val_accuracy'], label='val acc')
plt.xlabel('Epoch');
plt.ylabel('Accuracy');
plt.legend()

plt.tight_layout()
plt.show()

# Lưu weights
model.save('audio_recog_cnn.h5')
