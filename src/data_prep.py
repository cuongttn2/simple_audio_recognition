import os
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf

# Bước 2: Download dataset

# Đường dẫn và URL dataset
DATA_DIR = 'data'
DATASET_NAME = 'mini_speech_commands'
ZIP_NAME = f'{DATASET_NAME}.zip'
ZIP_URL = 'http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip'

dataset_path = os.path.join(DATA_DIR, DATASET_NAME)

# Nếu chưa tồn tại, download và giải nén dataset
if not os.path.isdir(dataset_path):
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, ZIP_NAME)
    print(f"Downloading {ZIP_URL} to '{zip_path}'...")
    urllib.request.urlretrieve(ZIP_URL, zip_path)
    print("Extracting zip file into data directory...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    os.remove(zip_path)
    print(f"Dataset extracted to '{dataset_path}'.")
else:
    print(f"Directory '{dataset_path}' already exists, skipping download.")

# Bước 3: Tạo train/val sets
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=dataset_path,
    batch_size=16,
    validation_split=0.2,
    output_sequence_length=16000,
    seed=0,
    subset='both'
)
label_names = np.array(train_ds.class_names)
print("Labels:", label_names)


# Squeeze extra channel
def squeeze(audio, labels):
    return tf.squeeze(audio, axis=-1), labels


train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
