import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.activations import relu, softmax
import os
import shutil


class DataGenerators():
    def __init__(self, train_path, test_path, enable_augmentation, preprocessing_fn):
        self.train_path = train_path
        self.test_path = test_path
        self.augmentation = enable_augmentation
        self.train_datagen = type(ImageDataGenerator)
        self.test_datagen = type(ImageDataGenerator)
        self.tensorboard_datagen = type(ImageDataGenerator)
        self.preprocessing_fn = preprocessing_fn

    def create_data_generators(self, size, rotation_range=0.3, width_shift_range=0.4, height_shift_range=0.4,
                               zoom_range=0.2):

        if self.augmentation:
            print("Augment!")
            self.train_datagen = ImageDataGenerator(rotation_range=rotation_range, width_shift_range=width_shift_range,
                                                    height_shift_range=height_shift_range,
                                                    zoom_range=zoom_range,
                                                    preprocessing_function=self.preprocessing_fn)
        else:
            self.train_datagen = ImageDataGenerator(preprocessing_function=self.preprocessing_fn)

        self.test_datagen = ImageDataGenerator(preprocessing_function=self.preprocessing_fn)

        self.tensorboard_datagen = ImageDataGenerator()

        self._create_generators(size)
        return self.train_generator, self.test_generator, self.tensorboard_generator

    def _create_generators(self, size):

        self.train_generator = self.train_datagen.flow_from_directory(self.train_path,
                                                                      target_size=(size, size),
                                                                      color_mode="rgb", batch_size=24)
        self.test_generator = self.test_datagen.flow_from_directory(self.test_path,
                                                                    target_size=(size, size),
                                                                    color_mode="rgb", batch_size=24)

        self.tensorboard_generator = self.tensorboard_datagen.flow_from_directory(self.test_path,
                                                                                  target_size=(size, size),
                                                                                  color_mode="rgb", batch_size=24)

    def get_samples(self):
        return self.train_generator.samples, self.test_generator.samples

    def get_batch_size(self):
        return self.train_generator.batch_size

    def get_label_mapping_dict(self):
        mapping_dict = {}
        for i, j in self.test_generator.class_indices.items():
            mapping_dict[j] = i
        return mapping_dict

    def get_image_shape(self):
        return self.train_generator.image_shape


def create_model(model, train_up_to=None, dense1=64):
    if train_up_to:
        set_untrainable(model, train_up_to)
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(dense1, activation=relu)(x)
    x = Dense(24, activation=softmax)(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def set_untrainable(model, up_to_what):
    for layer in model.layers:
        layer.trainable = True

    if up_to_what > 0:
        for layer in model.layers[:-up_to_what]:
            if layer.__class__.__name__ != "BatchNormalization":
                layer.trainable = False

    elif up_to_what == 0:
        for layer in model.layers:
            if layer.__class__.__name__ != "BatchNormalization":
                layer.trainable = False
    else:
        for layer in model.layers[:up_to_what]:
            if layer.__class__.__name__ != "BatchNormalization":
                layer.trainable = False


def top_3_acc(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, 3)


def ask_what_to_do_with_logfiles(log_dir, ckpt):
    log_dir = os.path.abspath(log_dir)
    key = input(f'Delete Logfiles in {log_dir}? (D)elete | K(eep) | Q(uit)')
    if (key == "d" or key == "D") and os.path.isdir(log_dir):
        for i in os.listdir(log_dir):
            path_to_delete = os.path.join(log_dir, i)
            if os.path.isdir(path_to_delete):
                shutil.rmtree(path_to_delete, ignore_errors=True)
            else:
                os.remove(path_to_delete)

    if key == "k" or key == "K":
        print(f'Keeping old Logfiles')

    if key == "q" or key == "Q":
        exit()

    key = input(f'Checkpoint dir correct? {ckpt}? Y | N)')
    if key == "Y" or key == "y":
        print(f'Start Training...')
    else:
        exit()
    return log_dir, ckpt
