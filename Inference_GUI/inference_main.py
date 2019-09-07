#!/usr/bin/env python3

import sys
import datetime
import logging
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSlot
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.uic import loadUi
import inference_gui_utils as gui_utils
from classification_utils import Inference


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class Menu(QDialog):

    def __init__(self):
        super().__init__()
        self.ui = loadUi("design.ui", self)
        self.ui.classify.setDisabled(True)

        self.predictor = Inference()

        self.image = QImage(self.size(), QImage.Format_Grayscale8)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brush_size = 2
        self.color = Qt.black
        self.last_point = QPoint()

        self.ui.clear.clicked.connect(self.clear_canvas)
        self.ui.classify.clicked.connect(self.classify_image)
        self.ui.select_model.clicked.connect(self.select_model_dialog)

        self.set_tooltips()

        self.show()

    @pyqtSlot()
    def select_model_dialog(self):
        gui_utils.set_widgets_to_gray(self)
        self.update()

        model_path, _ = QFileDialog.getOpenFileName(self)

        if model_path:
            gui_utils.display_model_path(self, model_path)
            self.predictor.load(model_path)
            logging.info("Model loaded")

        gui_utils.set_widgets_clickable(self)

    @pyqtSlot()
    def mousePressEvent(self, event):

        if (event.button() == Qt.LeftButton) & gui_utils.check_boundaries(self, event):
            self.drawing = True
            self.last_point = event.pos()

    @pyqtSlot()
    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & gui_utils.check_boundaries(self, event):
            painter = QPainter(self.image)
            painter.setPen(QPen(self.color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    @pyqtSlot()
    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    @pyqtSlot()
    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    @pyqtSlot()
    def clear_canvas(self):
        logging.debug("Clear canvas procedure executed.")
        self.image.fill(Qt.white)
        self.update()

    @pyqtSlot()
    def keyPressEvent(self, event):

        if event.key() == Qt.Key_Shift:
            self.clear_canvas()

        if event.key() == Qt.Key_P:
            self.classify_image()

        if event.key() == Qt.Key_Escape:
            logging.debug("Escape button pressed, terminate program.")
            logging.debug("Classification creation program terminated -- %s" % (datetime.datetime.now()))
            sys.exit(0)

    @pyqtSlot()
    def closeEvent(self, event):
        logging.debug("Classification program terminated -- %s" % (datetime.datetime.now()))

    @pyqtSlot()
    def classify_image(self):
        img = self.image.copy(QRect(70, 130, 224, 224))
        image_path = gui_utils.save_input_image(img)
        results = self.predictor.predict_top3(image_path)
        gui_utils.display_results(self, results)
        logging.info(f'Image classified as {results[0]} with {[results[1]]}')
        self.update()
        return

    @pyqtSlot()
    def set_tooltips(self):
        self.ui.label_model_selection.setToolTip("Select the pretrained model that shall"
                                                 " be used to classify the input image.")
        self.ui.label_result_first.setToolTip("Likelihood for the most probable classification"
                                              " result of the input image.")
        self.ui.label_result_second.setToolTip("Likelihood for the second most probable classification"
                                               " result of the input image.")
        self.ui.label_result_third.setToolTip("Likelihood for the third most probable classification result"
                                              " of the input image.")
        self.ui.label_results.setToolTip("The results of the image classification using the selected model"
                                         " in the table on the left.")

        self.ui.select_model.setToolTip("Select the pretrained model that shall be used to classify the input image.")

        self.ui.clear.setToolTip("Clear the drawing canvas above. Use SHIFT alternatively.")

        self.ui.classify.setToolTip("Click in this button to classify the drawn input image using"
                                    " the currently selected model in the table above.")

        self.ui.frame.setToolTip("Please draw the greek letter you want to classify.\nBoth,"
                                 " lower and uppercase letters work.")
        self.ui.result_first.setToolTip("The most probable classification result of the input image.")
        self.ui.result_second.setToolTip("The second most probable classification result of the input image.")
        self.ui.result_third.setToolTip("The third most probable classification result of the input image.")


if __name__ == '__main__':
    parser = gui_utils.argparse_parse()
    args = parser.parse_args()
    ARGS = args

    if args.debug:
        logging.basicConfig(filename='output.log',
                            filemode='w',
                            level=logging.DEBUG)
        logging.debug("Classification creation program started -- %s\n" % (datetime.datetime.now()))
    if not(args.debug):
        logging.info("Classification creation program started -- %s\n" % (datetime.datetime.now()))

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)

    app = QApplication(sys.argv)
    mainMenu = Menu()
    sys.exit(app.exec_())
