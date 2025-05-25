import os
import tensorflow as tf
import numpy as np

# Bước 2: Download dataset
data = tf.keras.utils.get_file(
    'mini_speech_commands.zip',
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    extract=True,
    cache_dir='.', cache_subdir='data')
# ./data/mini_speech_commands/

# Bước 3: Tạo train/val sets
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory='./data/mini_speech_commands',
    batch_size=16,
    validation_split=0.2,
    output_sequence_length=16000,
    seed=0,
    subset='both')
label_names = np.array(train_ds.class_names)
print("Labels:", label_names)

# Squeeze extra channel
def squeeze(audio, labels):
    return tf.squeeze(audio, axis=-1), labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds   = val_ds.map(squeeze, tf.data.AUTOTUNE)
