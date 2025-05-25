import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from train import test_set, model  # giả sử model đã load

# Predict và tính confusion matrix
y_pred = np.argmax(model.predict(test_set), axis=1)
y_true = np.concatenate([y for x,y in test_set], axis=0)
cm = tf.math.confusion_matrix(y_true, y_pred)

# Vẽ heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.show()

# Classification report
print(classification_report(y_true, y_pred))
