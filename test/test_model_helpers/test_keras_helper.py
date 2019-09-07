import sys
sys.path.append("..")
from context import set_untrainable, DataGenerators, create_model, top_3_acc  # noqa
import tensorflow as tf  # noqa
import tensorflow.keras.backend as K  # noqa
import numpy as np  # noqa
import os  # noqa


def get_trainable_layers(model):
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))

    return trainable_count


def create_dense_model(input_size, d1, d2):

    inputs = tf.keras.layers.Input(shape=input_size)
    x = tf.keras.layers.Dense(d1)(inputs)
    x = tf.keras.layers.Dense(d2)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    return model


def create_cnn_model(input_shape, f1, f2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(f1, (3, 3))(inputs)
    x = tf.keras.layers.Conv2D(f2, (3, 3))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


def base_model(input_shape, f1, f2):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(f1, (3, 3))(inputs)
    x = tf.keras.layers.Conv2D(f2, (3, 3))(x)
    return tf.keras.models.Model(inputs=inputs, outputs=x)


def mock_preprocessing_fn(x):
    return x


path = os.path.dirname(os.path.abspath(__file__))
empty_path = os.path.join(path, "test_files/empty_tests/train")
images_path = os.path.join(path, "test_files/images")

alphabetic_label_mapping = {0: 'Alpha', 1: 'Beta', 2: 'Chi', 3: 'Delta', 4: 'Epsilon', 5: 'Eta', 6: 'Gamma', 7: 'Iota',
                            8: 'Kappa', 9: 'Lambda', 10: 'My', 11: 'Ny', 12: 'Omega', 13: 'Omikron', 14: 'Phi',
                            15: 'Pi', 16: 'Psi', 17: 'Rho', 18: 'Sigma', 19: 'Tau', 20: 'Theta', 21: 'Xi',
                            22: 'Ypsilon', 23: 'Zeta'}


def test_get_trainable_layers():
    model = create_dense_model((10,), 10, 10)
    assert get_trainable_layers(model) == 220


def test_set_untrainable_on_dense1():
    model = create_dense_model((10,), 10, 10)
    set_untrainable(model, 1)
    assert get_trainable_layers(model) == 110


def test_set_untrainable_on_dense2():
    model = create_dense_model((10,), 10, 10)
    set_untrainable(model, 0)
    assert get_trainable_layers(model) == 0


def test_get_trainable_layers2():
    model = create_dense_model((33,), 27, 41)
    assert get_trainable_layers(model) == 2066


def test_set_untrainable_on_dense3():
    model = create_dense_model((33,), 27, 41)
    set_untrainable(model, 1)
    assert get_trainable_layers(model) == 1148


def test_set_untrainable_on_dense4():
    model = create_dense_model((33,), 27, 41)
    set_untrainable(model, 0)
    assert get_trainable_layers(model) == 0


def test_get_trainable_layers3():
    model = create_cnn_model((28, 28, 1), 10, 10)
    assert get_trainable_layers(model) == 100+910+5761


def test_set_untrainable_on_cnn1():
    model = create_cnn_model((28, 28, 1), 10, 10)
    set_untrainable(model, 1)
    assert get_trainable_layers(model) == 5761


def test_set_untrainable_on_cnn2():
    model = create_cnn_model((28, 28, 1), 10, 10)
    set_untrainable(model, 3)
    assert get_trainable_layers(model) == 5761 + 910


def test_set_untrainable_on_cnn3():
    model = create_cnn_model((28, 28, 1), 10, 10)
    set_untrainable(model, 4)
    assert get_trainable_layers(model) == 5761 + 910 + 100


def test_set_untrainable_on_cnn4():
    model = create_cnn_model((28, 28, 1), 10, 10)
    set_untrainable(model, 0)
    assert get_trainable_layers(model) == 0


def test_set_untrainable_with_negative_argument():
    model = create_cnn_model((28, 28, 1), 10, 10)
    set_untrainable(model, -1)
    assert get_trainable_layers(model) == 5761


def test_Data_Generator():
    dg = DataGenerators(images_path, images_path, True, mock_preprocessing_fn)
    dg.create_data_generators(299)
    train_samples, test_samples = dg.get_samples()
    assert train_samples == test_samples == 11


def test_Data_Generator_no_Augmentation():
    dg = DataGenerators(images_path, images_path, False, mock_preprocessing_fn)
    dg.create_data_generators(299)
    assert dg.augmentation is False


def test_Data_Generator_batch_size():
    dg = DataGenerators(images_path, images_path, True, mock_preprocessing_fn)
    dg.create_data_generators(299)
    batch_size = dg.get_batch_size()
    assert batch_size == 24


def test_Data_Generator_label_mapping():
    dg = DataGenerators(images_path, images_path, True, mock_preprocessing_fn)
    dg.create_data_generators(299)
    label_mapping = dg.get_label_mapping_dict()
    print(label_mapping)
    assert label_mapping == alphabetic_label_mapping


def test_Data_Generator_image_shape():
    dg = DataGenerators(images_path, images_path, True, mock_preprocessing_fn)
    dg.create_data_generators(299)
    shape = dg.get_image_shape()
    assert shape == (299, 299, 3)


def test_create_model():
    mock_model = base_model((28, 28, 3), 10, 10)
    model = create_model(mock_model, train_up_to=0)
    assert get_trainable_layers(model) == (280+910) + (10*64)+64 + (64*24)+24


def test_create_model_with_frozen_weights():
    mock_model = base_model((28, 28, 3), 10, 10)
    model = create_model(mock_model, train_up_to=1)
    assert get_trainable_layers(model) == 910 + (10 * 64) + 64 + (24*64) + 24


def test_top_3_acc():
    y_true = tf.constant(np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]), dtype=tf.int32)
    y_pred = tf.constant(np.array([[0.3, 0.4, 0.2, 0.1],
                                   [0.3, 0.4, 0.2, 0.1],
                                   [0.3, 0.4, 0.2, 0.1],
                                   [0.3, 0.4, 0.2, 0.1]]), dtype=tf.float32)

    with tf.Session() as sess:
        assert sess.run(top_3_acc(y_true, y_pred)) == 0.75


def test_top_3_acc_all():
    y_true = tf.constant(np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]), dtype=tf.int32)
    y_pred = tf.constant(np.array([[0.3, 0.4, 0.2, 0.1],
                                   [0.3, 0.4, 0.2, 0.1],
                                   [0.3, 0.4, 0.2, 0.1],
                                   [0.3, 0.3, 0.15, 0.25]]), dtype=tf.float32)

    with tf.Session() as sess:
        assert sess.run(top_3_acc(y_true, y_pred)) == 1
