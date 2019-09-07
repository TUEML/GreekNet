import os
import argparse
from PyQt5.QtGui import QPixmap


SELECTION_STOP = 0
SELECTION_GO = 1

ABS_PATH = os.path.abspath(os.path.dirname(__file__))
GROUND_TRUTH_PATH = os.path.join(ABS_PATH, "../Ground_Truth")


def check_boundaries(self, forwarded_event):
    x = self.ui.frame.x()
    y = self.ui.frame.y()
    x2 = x + self.ui.frame.width()
    y2 = y + self.ui.frame.height()

    if (x < forwarded_event.x() < x2) & (y < forwarded_event.y() < y2):
        return True
    return False


def display_model_path(self, model_path):
    self.ui.label_model.setText(model_path)


def display_results(self, results):
    global GROUND_TRUTH_PATH
    for inx, res in enumerate(results[0]):
        postfix = ".png"
        filename = ("%s%s" % (res, postfix))
        filename = os.path.join(GROUND_TRUTH_PATH, filename)
        ground_truth = load_ground_truth(filename)
        if inx == 0:
            self.ui.result_first.setPixmap(ground_truth)
            val = str(round(results[1][0], 4))
            new_label_text = f'1 - {val}'
            self.ui.label_result_first.setText(new_label_text)
        if inx == 1:
            self.ui.result_second.setPixmap(ground_truth)
            val = str(round(results[1][1], 4))
            new_label_text = f'2 - {val}'
            self.ui.label_result_second.setText(new_label_text)
        if inx == 2:
            self.ui.result_third.setPixmap(ground_truth)
            val = str(round(results[1][2], 4))
            new_label_text = f'3 - {val}'
            self.ui.label_result_third.setText(new_label_text)


def load_ground_truth(filename):
    pixmap = QPixmap(filename)
    return pixmap


def save_input_image(image):
    global ABS_PATH
    filename = os.path.join(ABS_PATH, "input.png")
    image.save(f'{filename}')
    return filename


def argparse_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--debug',
                        help='Flag to enable verbose debug output.',
                        action='store_true')
    return parser


def set_widgets_to_gray(self):
    self.ui.clear.setDisabled(True)
    self.ui.classify.setDisabled(True)
    self.ui.select_model.setDisabled(True)


def set_widgets_clickable(self):
    self.ui.clear.setDisabled(False)
    self.ui.classify.setDisabled(False)
    self.ui.select_model.setDisabled(False)
