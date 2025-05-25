import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report
from data_prep import dataset_path
from data_prep import squeeze
from data_prep import train_ds, val_ds, label_names

# 1) Load mô hình đã train
model = tf.keras.models.load_model('audio_recog_cnn.keras')  # hoặc .h5 nếu bạn dùng h5

# 2) Tạo test_set lại giống train.py
def get_spectrogram(waveform):
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    return tf.abs(spec)[..., tf.newaxis]
def to_spec(ds):
    # ds đã là (waveform, label) với shape [batch,16000]
    return ds.map(lambda x, y: (get_spectrogram(x), y),
                  num_parallel_calls=tf.data.AUTOTUNE)

# Tải lại toàn bộ dữ liệu train/val
train_ds_full, val_ds_full = tf.keras.utils.audio_dataset_from_directory(
    directory=dataset_path,
    batch_size=16,
    validation_split=0.2,
    output_sequence_length=16000,
    seed=0,
    subset='both'
)
# Loại bỏ extra channel
train_ds_full = train_ds_full.map(squeeze, tf.data.AUTOTUNE)
val_ds_full   = val_ds_full.map(squeeze,   tf.data.AUTOTUNE)

# Chuyển thành spectrogram và tách test_set
val_spec = to_spec(val_ds_full)
val_size = val_spec.cardinality() // 2
test_set = val_spec.skip(val_size)

# 3) Predict và tính metrics
y_pred = np.argmax(model.predict(test_set), axis=1)
y_true = np.concatenate([y for x,y in test_set], axis=0)

# Confusion matrix
cm = tf.math.confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_true, y_pred, target_names=label_names))
