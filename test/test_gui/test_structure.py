import sys
sys.path.append("..")
import os  # noqa
import shutil  # noqa
from context import directory_management  # noqa

path = os.path.dirname(os.path.abspath(__file__))
top_level = os.path.join(path, "top")

images_path = os.path.join(top_level, "train")

GREEK_ALPHABET = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "My",
                  "Ny", "Xi", "Omikron", "Pi", "Rho", "Sigma", "Tau", "Ypsilon", "Phi", "Chi", "Psi", "Omega"]


class MocKUi:
    def __init__(self):
        self.top_directory = top_level
        self.train_directory = os.path.join(top_level, "train")
        self.test_directory = os.path.join(top_level, "test")


def teardown_function():
    if os.path.isdir(top_level):
        shutil.rmtree(top_level)


def test_create_top_level():
    directory_management.create_directories(MocKUi())
    assert "top" in os.listdir(path)


def test_structure_greek_alphabet():
    directory_management.create_directories(MocKUi())
    dir_list = os.listdir(images_path)
    for i in dir_list:
        assert i in GREEK_ALPHABET
