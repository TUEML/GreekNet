from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt

model = load_model("Pretrained_with_DA/checkpoints/resnet_final/200.ckpt", compile=False)

print(model.summary())

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = test_datagen.flow_from_directory("../Dataset/test", target_size=(224, 224), batch_size=576)


def get_label_mapping_dict():
    mapping_dict = {}
    for i, j in test_data.class_indices.items():
        mapping_dict[j] = i
    return mapping_dict


mapping_dict = get_label_mapping_dict()
print(mapping_dict)

X_test, y_test = next(test_data)


incorrect = 0
for X, y in zip(X_test, y_test):
    pred = np.argsort(model.predict(X.reshape(1, 224, 224, 3)))[0][-3:][::-1]
    if not y.argmax() in pred:
        mapped_preds = [mapping_dict[x] for x in pred]
        print(mapped_preds)
        incorrect += 1
        plt.figure()
        plt.imshow(X)
        plt.title(mapped_preds)


print(incorrect)

plt.show()
