from tflite_support.metadata_writers import audio_classifier
from tflite_support.metadata_writers import writer_utils

# File đã convert từ keras sang tflite
MODEL_PATH = "converted_model.tflite"
LABEL_FILE = "labels.txt"

# File mới sẽ sinh ra
OUTPUT_MODEL = "converted_model_with_metadata.tflite"

# Cấu hình đầu vào: bạn dùng 1 giây audio mono, sample_rate 16kHz
SAMPLE_RATE = 16000
CHANNELS = 1
INPUT_TENSOR_LENGTH = 16000  # 1 giây * 16000 Hz

# Tạo metadata writer
writer = audio_classifier.AudioClassifierWriter.create_from_file(
    MODEL_PATH,
    LABEL_FILE,
    sample_rate=SAMPLE_RATE,
    channels=CHANNELS,
    input_tensor_length=INPUT_TENSOR_LENGTH
)

# Ghi ra file mới có metadata
writer_utils.save_file(writer.populate(), OUTPUT_MODEL)
print(f" Done: created {OUTPUT_MODEL}")
