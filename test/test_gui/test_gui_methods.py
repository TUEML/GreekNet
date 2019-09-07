import sys
sys.path.append("..")
import os # noqa
from context import gui_utils  # noqa
from context import main  # noqa
from PyQt5.QtWidgets import QDialog, QApplication  # noqa
import statistic_utils  # noqa

path = os.path.dirname(os.path.abspath(__file__))
top_level = os.path.join(path, "top")

SELECTION_STOP = 0
SELECTION_GO = 1

parser = statistic_utils.argparse_parse()
args = parser.parse_args(["-sf", os.path.join(path, "test_files/dataset_statistics.csv")])
app = QApplication(sys.argv)
menu = main.Menu(args)


class MockEvent:
    def __init__(self, x, y):
        self.x_val = x
        self.y_val = y

    def x(self):
        return self.x_val

    def y(self):
        return self.y_val


# boundaries are 224, 224. x is 70, y is 130
def test_check_boundaries_correct():
    assert gui_utils.check_boundaries(menu, MockEvent(200, 200))


def test_check_boundaries_border():
    assert gui_utils.check_boundaries(menu, MockEvent(69+224, 129+224))


def test_check_boundaries_out_of_canvas():
    assert gui_utils.check_boundaries(menu, MockEvent(350, 200)) is False


def test_check_boundaries_out_of_canvas2():
    assert gui_utils.check_boundaries(menu, MockEvent(200, 350)) is True


def test_check_boundaries_out_of_canvas3():
    assert gui_utils.check_boundaries(menu, MockEvent(350, 350)) is False


def test_check_boundaries_out_of_canvas4():
    assert gui_utils.check_boundaries(menu, MockEvent(69, 200)) is False


def test_check_boundaries_out_of_canvas5():
    assert gui_utils.check_boundaries(menu, MockEvent(200, 79)) is False


def test_check_boundaries_out_of_canvas6():
    assert gui_utils.check_boundaries(menu, MockEvent(0, 0)) is False


def test_select_top_directory_path_init():
    assert menu.ui.save.isEnabled() is False


def test_select_top_directory_path():
    test_path = os.path.join(path, "test_files/top")

    menu.creation_mode = 0
    menu.train_directory = os.path.join(test_path, "train")
    menu.test_directory = os.path.join(test_path, "test")
    menu.top_directory = top_level
    gui_utils.select_top_directory_path(menu)
    assert menu.ui.storage_directory.text() == menu.train_directory


def test_count_images():
    test_path = os.path.join(path, "test_files/images/Alpha")
    assert gui_utils.count_images(test_path) == 2


def test_get_absolute_file_name():

    test_path = os.path.join(path, "test_files/images")
    menu.train_directory = test_path

    fname = gui_utils.get_absolute_file_name(menu, "Alpha")
    assert fname == f'{test_path}/Alpha/3_Alpha.png'


def test_scan_directory_for_preexisting_data():
    test_path = os.path.join(path, "test_files/images")
    assert gui_utils.scan_directory_for_preexisting_data(test_path) is True


def test_no_greek_symbol_in_dirs():
    test_path = os.path.join(path, "test_files")
    assert gui_utils.scan_directory_for_preexisting_data(test_path) is False
