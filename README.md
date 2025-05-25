# Audio Recognition in TensorFlow

Mô tả: project nhận dạng 8 từ (`yes`, `no`, …) từ dataset mini_speech_commands.

## Cài đặt
1. `python3.10 -m venv .venv && source .venv/bin/activate`  
2. `pip install -r requirements.txt`  

## Chạy
- Chuẩn bị & load dữ liệu:  
  `python src/data_prep.py`
- Huấn luyện:  
  `python src/train.py`
- Đánh giá:  
  `python src/evaluate.py`
- Dự đoán mẫu mới:  
  `python src/inference.py`
