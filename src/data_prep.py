import os
import urllib.request
import zipfile

import numpy as np
import tensorflow as tf

# Bước 2: Download dataset

# Comment lại khi train cho câu lệnh riêng =====>
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
    print("Extracting zip file into data directory (skipping macOS metadata)...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.infolist():
            filename = member.filename
            # Bỏ qua thư mục __MACOSX và file .DS_Store
            if filename.startswith('__MACOSX') or os.path.basename(filename) == '.DS_Store':
                continue
            zip_ref.extract(member, DATA_DIR)
    os.remove(zip_path)
    print(f"Dataset extracted to '{dataset_path}'.")
else:
    print(f"Directory '{dataset_path}' already exists, skipping download.")

# Comment lại khi train cho câu lệnh riêng <=====

# Dùng code sau để train câu lệnh riêng ==>
# KEYWORDS = ['hello', 'bye']
# DATA_DIR = 'data/commands' # thư mục customize câu lệnh riêng ('hello', 'bye')

# Bước 3: Tạo train/val sets

# Comment lại khi train cho câu lệnh riêng =====>
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=dataset_path,
    batch_size=16,
    validation_split=0.2,
    output_sequence_length=16000,
# Set the seed value for experiment reproducibility.
    seed=0,
    subset='both'
)
# Comment lại khi train cho câu lệnh riêng <=====

# Dùng code sau để train câu lệnh riêng ==>
# train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
#     directory=DATA_DIR,
#     labels='inferred',
#     label_mode='int',
#     class_names=KEYWORDS,
#     batch_size=16,
#     validation_split=0.2,
#     subset='both',
#     seed=42,
#     output_sequence_length=16000
# )

label_names = np.array(train_ds.class_names)
print("Labels:", label_names)

with open("labels.txt", "w") as f:
    for label in train_ds.class_names:
        f.write(label + "\n")

# Squeeze extra channel
def squeeze(audio, labels):
    return tf.squeeze(audio, axis=-1), labels


train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)
