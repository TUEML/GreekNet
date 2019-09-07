import sys
import os
sys.path.append("..")
from context import Inference  # noqa
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


path = os.path.dirname(os.path.abspath(__file__))


def test_load_model():
    model_path = os.path.join(path, "vgg16/Model_224.ckpt")
    model = Inference().load(model_path)
    assert model.layers[0].output_shape == (None, 224, 224, 3)


def test_load_model_299():
    model_path = os.path.join(path, "inception_v3/Model_299.ckpt")
    model = Inference().load(model_path)
    assert model.layers[0].output_shape == (None, 299, 299, 3)


def test_predict_top3():
    model_path = os.path.join(path, "vgg16/Model_224.ckpt")
    image_path = os.path.join(path, "1_Alpha.png")
    classifier = Inference()
    classifier.load(model_path)
    top3, _ = classifier.predict_top3(image_path)
    assert list(top3) == ["Alpha", "Chi", "Zeta"]


def test_predict_top3_299():
    model_path = os.path.join(path, "inception_v3/Model_299.ckpt")
    image_path = os.path.join(path, "1_Alpha.png")
    classifier = Inference()
    classifier.load(model_path)
    top3, _ = classifier.predict_top3(image_path)
    assert list(top3) == ["Alpha", "Gamma", "Omega"]
