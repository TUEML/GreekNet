#!/usr/bin/env python3

import sys
import os
import datetime
import logging
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSlot
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.uic import loadUi
import gui_utils
import statistic_utils


TRAIN = 0
TEST = 1


class Menu(QDialog):

    def __init__(self, statistic_args):
        super().__init__()
        path = os.path.abspath(os.path.dirname(__file__))
        design = os.path.join(path, "design.ui")
        self.ui = loadUi(design, self)
        self.args = statistic_args
        self.statistics = statistic_utils.Statistic(statistic_args)

        self.image = QImage(self.size(), QImage.Format_Grayscale8)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brush_size = 2
        self.color = Qt.black

        self.creation_mode = TRAIN

        self.last_point = QPoint()

        self.ui.save.clicked.connect(self.save_image_according_to_label)
        self.ui.clear.clicked.connect(self.clear_canvas)
        self.ui.FileDialog.clicked.connect(self.open_file_dialog)
        self.ui.storage_selection_default.clicked.connect(self.set_default_storage_path)
        self.ui.radio_btn_train.clicked.connect(self.set_to_train)
        self.ui.radio_btn_test.clicked.connect(self.set_to_test)
        self.ui.btn_show_statistic.clicked.connect(self.show_dataset_statistic)
        self.ui.tableWidget.itemSelectionChanged.connect(self.update_lcd_statistic)
        self.ui.tableWidget.setCurrentCell(0, 0)

        gui_utils.enable_default_storage_directory()
        self.show()

    @property
    def creation_mode(self):
        return self.__creation_mode

    @creation_mode.setter
    def creation_mode(self, value):
        global TRAIN, TEST
        if value != TRAIN and value != TEST:
            self.__creation_mode = TRAIN
        else:
            self.__creation_mode = value

    @property
    def top_directory(self):
        return self.__top_directory

    @top_directory.setter
    def top_directory(self, value):
        if value == "":
            current_working_directory = os.getcwd()
            self.__top_directory = current_working_directory
        elif value != "" and len(value) < 2:
            self.__top_directory = current_working_directory
        else:
            self.__top_directory = value

    @property
    def train_directory(self):
        return self.__train_directory

    @train_directory.setter
    def train_directory(self, value):
        if value == "" and self.top_directory != "":
            self.__train_directory = ("%s%s" % (self.top_directory, "/train"))
        elif value != "" and len(value) < 4 and self.top_directory != "":
            self.__train_directory = ("%s%s" % (self.top_directory, "/train"))
        else:
            self.__train_directory = value

    @property
    def test_directory(self):
        return self.__test_directory

    @test_directory.setter
    def test_directory(self, value):
        if value == "" and self.top_directory != "":
            self.__test_directory = ("%s%s" % (self.top_directory, "/test"))
        elif value != "" and len(value) < 4 and self.top_directory != "":
            self.__test_directory = ("%s%s" % (self.top_directory, "/test"))
        else:
            self.__test_directory = value

    @pyqtSlot()
    def open_file_dialog(self):

        fname = QFileDialog.getExistingDirectory(self, "Select top level directory for saving the dataset images")
        self.top_directory = fname
        self.train_directory = ("%s%s" % (self.top_directory, "/train"))
        self.test_directory = ("%s%s" % (self.top_directory, "/test"))

        self.statistics.collect_statistics_from_storage_directory(ARGS, self.train_directory, self.test_directory)
        self.update_lcd_statistic()

        gui_utils.confirm_select_top_directory_path(self)

    @pyqtSlot()
    def set_default_storage_path(self):

        current_directory = os.path.split(os.getcwd())

        if current_directory[1] == "Dataset":
            self.top_directory = os.getcwd()
            self.train_directory = ("%s%s" % (self.top_directory, "/train"))
            self.test_directory = ("%s%s" % (self.top_directory, "/test"))
            gui_utils.select_top_directory_path(self)

            self.statistics.collect_statistics_from_storage_directory(self.args, self.train_directory,
                                                                      self.test_directory)
            self.update_lcd_statistic()
        else:
            QMessageBox.warning(self,
                                "Error using Default Path",
                                "The default storage directory 'Dataset' could not be selected.\n\n"
                                "Please use the 'Select Storage Directory' function to manually select a"
                                " storage directory.")

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
    def save_image_according_to_label(self):
        if gui_utils.check_image_changed(self):
            row = self.ui.tableWidget.currentRow()
            col = self.ui.tableWidget.currentColumn()
            letter = self.ui.tableWidget.item(row, col).text()
            cropped = self.image.copy(QRect(70, 130, 224, 224))

            filename = gui_utils.get_absolute_file_name(self, letter)

            cropped.save(f'{filename}')
            logging.debug("Save image on disk: %s" % (filename))
            self.statistics.collect_statistics_from_storage_directory(self.args, self.train_directory,
                                                                      self.test_directory)
            logging.debug("Updated statistics object by scanning the dataset directory on disk.")

            self.clear_canvas()
        else:
            logging.debug("Empty image detected, do not save it on disk.")
            pass
        self.update_lcd_statistic()

    @pyqtSlot()
    def clear_canvas(self):
        logging.debug("Clear canvas procedure executed.")
        self.image.fill(Qt.white)
        self.update()

    @pyqtSlot()
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Alt:
            try:
                if len(self.top_directory) > 0:
                    self.save_image_according_to_label()
            except (FileNotFoundError, AttributeError):
                logging.debug("Saving picture executed via ALT but no storage directory has been defined.")
                QMessageBox.warning(self,
                                    "Error saving picture",
                                    "No storage directory has been selected so far.\n"
                                    "Use the buttons on the right to select a storage directory for your dataset.")
                pass

        if event.key() == Qt.Key_Shift:
            self.clear_canvas()

        if event.key() == Qt.Key_F2:
            self.switch_creation_mode()

        if event.key() == Qt.Key_Escape:
            logging.debug("Escape button pressed, terminate program.")
            try:
                if len(self.top_directory) > 0:
                    self.statistics.update_statistics_file(self.top_directory, self.args)
            except (FileNotFoundError, AttributeError):
                QMessageBox.warning(self,
                                    "Error saving dataset statistics",
                                    "No storage directory has been selected so far.\n"
                                    "Use the buttons on the right to select a storage directory for your dataset.")
                pass
            logging.debug("Dataset creation program terminated -- %s" % (datetime.datetime.now()))
            sys.exit(0)

    @pyqtSlot()
    def switch_creation_mode(self):
        global TRAIN, TEST

        if self.ui.radio_btn_train.isChecked():
            self.ui.radio_btn_train.setChecked(False)
            self.ui.radio_btn_test.setChecked(True)
            self.creation_mode = TEST
            self.update_lcd_statistic()
            try:
                self.ui.storage_directory.setText(self.test_directory)
            except AttributeError:
                pass
            return

        if self.ui.radio_btn_test.isChecked():
            self.ui.radio_btn_test.setChecked(False)
            self.ui.radio_btn_train.setChecked(True)
            self.creation_mode = TRAIN
            self.update_lcd_statistic()
            try:
                self.ui.storage_directory.setText(self.train_directory)
            except AttributeError:
                pass
            return

    @pyqtSlot()
    def set_to_train(self):
        global TRAIN

        self.creation_mode = TRAIN
        try:
            self.ui.storage_directory.setText(self.train_directory)
        except AttributeError:
            logging.debug("Attempt to set data creation mode to TRAIN but no storage directory was selected.")
            pass

    @pyqtSlot()
    def set_to_test(self):
        global TEST

        self.creation_mode = TEST
        try:
            self.ui.storage_directory.setText(self.test_directory)
        except AttributeError:
            logging.debug("Attempt to set data creation mode to TEST but no storage directory was selected.")
            pass

    @pyqtSlot()
    def show_dataset_statistic(self):
        try:
            if len(self.top_directory) > 0:
                self.statistics.collect_statistics_from_storage_directory(self.args, self.train_directory,
                                                                          self.test_directory)
                dataset_statistic = self.statistics.get_statistic()
                gui_utils.statistic_message_box(self, dataset_statistic)
        except (FileNotFoundError, AttributeError):
            logging.debug("Show dataset statistic - no storage directory selected.")
            QMessageBox.warning(self,
                                "Error displaying statistics",
                                "No storage directory has been selected so far.\n"
                                "Use the buttons on the right to select a storage directory for your dataset.")
            pass

    @pyqtSlot()
    def closeEvent(self, event):
        try:
            self.statistics.update_statistics_file(self.top_directory, self.args)
        except AttributeError:
            logging.debug("Close event - no dataset storage path set.")
            pass
        logging.debug("Dataset creation program terminated -- %s" % (datetime.datetime.now()))

    @pyqtSlot()
    def update_lcd_statistic(self):
        statistics = self.statistics.get_statistic()

        if self.ui.radio_btn_train:
            row = self.ui.tableWidget.currentRow()
            col = self.ui.tableWidget.currentColumn()
            letter = self.ui.tableWidget.item(row, col).text()
            statistic_train = statistics[0]

            for value in statistic_train:
                if value == letter:
                    self.ui.lcd_samples_train.display(statistic_train[value])

        if self.ui.radio_btn_test:
            row = self.ui.tableWidget.currentRow()
            col = self.ui.tableWidget.currentColumn()
            letter = self.ui.tableWidget.item(row, col).text()
            statistic_test = statistics[1]

            for value in statistic_test:
                if value == letter:
                    self.ui.lcd_samples_test.display(statistic_test[value])


if __name__ == '__main__':
    parser = statistic_utils.argparse_parse()
    args = parser.parse_args()
    ARGS = args

    if args.debug:
        logging.basicConfig(filename='output.log',
                            filemode='w',
                            level=logging.DEBUG)
        logging.debug("Dataset creation program started -- %s\n" % (datetime.datetime.now()))
    if not(args.debug):
        logging.info("Dataset creation program started -- %s\n" % (datetime.datetime.now()))

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        logging.debug("Use statistics file: %s" % (args.statistics_filename))

    app = QApplication(sys.argv)
    mainMenu = Menu(args)
    sys.exit(app.exec_())
