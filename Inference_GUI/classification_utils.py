from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.applications.resnet50 import preprocess_input as resnet_50_preprocess_fn
from tensorflow.python.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_fn
from tensorflow.python.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_fn
from tensorflow.python.keras.applications.inception_v3 import preprocess_input as inception_preprocess_fn
import cv2
import numpy as np


LABEL_MAPPING_DICT = {0: 'Alpha', 1: 'Beta', 2: 'Chi', 3: 'Delta', 4: 'Epsilon', 5: 'Eta', 6: 'Gamma', 7: 'Iota',
                      8: 'Kappa', 9: 'Lambda', 10: 'My', 11: 'Ny', 12: 'Omega', 13: 'Omikron', 14: 'Phi', 15: 'Pi',
                      16: 'Psi', 17: 'Rho', 18: 'Sigma', 19: 'Tau', 20: 'Theta', 21: 'Xi', 22: 'Ypsilon', 23: 'Zeta'}


class Inference():

    def __init__(self):
        self.input_shape = (None, 224, 224, 3)
        self.model = None
        self.preprocess_fn = None

    def load(self, path_to_model):
        model = load_model(path_to_model, compile=False)
        self.input_shape = model.layers[0].output_shape
        self.model = model
        self.preprocess_fn = self._get_correct_preprocessing_function(path_to_model)
        return model

    def predict_top3(self, image_path):
        image = cv2.imread(image_path)

        preprocessed_image = image.copy()
        preprocessed_image = cv2.resize(preprocessed_image, (self.input_shape[1], self.input_shape[2]))
        preprocessed_image = preprocessed_image.reshape(1, self.input_shape[1], self.input_shape[2],
                                                        self.input_shape[3])

        preprocessed_image = self.preprocess_fn(preprocessed_image)
        pred = self.model.predict(preprocessed_image)

        top_3 = np.argsort(pred)[0][-3:][::-1]
        top_3_percentage = np.sort(pred)[0][-3:][::-1]

        top_3_greek = [LABEL_MAPPING_DICT[i] for i in top_3]

        return top_3_greek, top_3_percentage*100

    def _get_correct_preprocessing_function(self, path):

        if "vgg16" in path:
            return vgg16_preprocess_fn

        elif "vgg19" in path:
            return vgg19_preprocess_fn

        elif "resnet" in path:
            return resnet_50_preprocess_fn

        elif "inception" in path:
            return inception_preprocess_fn
