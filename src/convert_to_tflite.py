import tensorflow as tf

# 1. Load model native Keras
model = tf.keras.models.load_model('audio_recog_cnn.keras')

# 2. Tạo converter từ Keras Model
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# (Tuỳ chọn) thêm quantization để giảm kích thước & tăng tốc
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 3. Convert
tflite_model = converter.convert()

# 4. Save file .tflite
with open('audio_recog_cnn.tflite', 'wb') as f:
    f.write(tflite_model)

print("Saved TFLite model to audio_recog_cnn.tflite")
