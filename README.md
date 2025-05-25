# Audio Recognition in TensorFlow

Mô tả: project nhận dạng các từ từ dataset `mini_speech_commands`.

## Yêu cầu

* Python 3.10 trở lên
* pip
* Virtual environment (`venv`)

## Thiết lập môi trường

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## `requirements.txt`

```text
tensorflow-macos>=2.12
tensorflow-metal
numpy
matplotlib
scikit-learn
```

## Cấu trúc project

```
audio_recognition_tensorflow/
├── data/
│   └── mini_speech_commands/
├── src/
│   ├── data_prep.py      # Tải và chuẩn hoá dữ liệu
│   ├── model.py          # Định nghĩa kiến trúc CNN
│   ├── train.py          # Huấn luyện, in shape, plot training curves
│   ├── evaluate.py       # Vẽ confusion matrix, classification report
│   ├── inference.py      # Demo dự đoán file WAV mới
│   └── visualize.py      # Vẽ waveform và spectrogram mẫu
├── requirements.txt
└── README.md
```

## Chạy project

1. **Chuẩn bị dữ liệu**

   ```bash
   python src/data_prep.py
   ```

2. **Huấn luyện**
   Mặc định `train.py` sử dụng `EPOCHS = 10` và lưu mô hình dạng native Keras (`.keras`).

   ```bash
   python src/train.py
   ```

3. **Đánh giá mô hình**

   ```bash
   python src/evaluate.py
   ```

   * Vẽ confusion matrix và classification report.

4. **Dự đoán mẫu mới**

   ```bash
   python src/inference.py
   ```

   * Load `audio_recog_cnn.keras` và dự đoán file WAV chỉ định trong script.

5. **Trực quan hoá dữ liệu**

   ```bash
   python src/visualize.py
   ```

   * Vẽ waveform và spectrogram của một sample đầu tiên từ `train_ds`, đồng thời phát âm thanh.

## Tùy chỉnh

* **Batch size**: chỉnh trong `src/data_prep.py` (tham số `batch_size`).
* **Epochs**: chỉnh biến `EPOCHS` trong `src/train.py`.
* **Callbacks**: thêm callback như `EarlyStopping` trong `train.py`.
* **Visualization**: tuỳ chỉnh sample index hoặc plot format trong `visualize.py`.

---

Chúc bạn thành công!
