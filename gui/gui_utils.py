import os
import logging
from PyQt5.QtWidgets import QMessageBox
import directory_management


SELECTION_STOP = 0
SELECTION_GO = 1


def check_boundaries(self, forwarded_event):
    x = self.ui.frame.x()
    y = self.ui.frame.y()
    x2 = x + self.ui.frame.width()
    y2 = y + self.ui.frame.height()

    if (x < forwarded_event.x() < x2) & (y < forwarded_event.y() < y2):
        return True
    return False


def confirm_select_top_directory_path(self):
    if len(self.top_directory) > 0 and check_for_existing_dataset(self):
        if confirmation_message_box(self):
            directory_management.create_directories(self)
            self.ui.save.setDisabled(False)
            if self.creation_mode == 0:
                logging.debug("Creation mode TRAIN, update UI labels.")
                self.ui.storage_directory.setText(self.train_directory)
                logging.debug("Set storage directory: %s" %
                              (self.train_directory))

            if self.creation_mode == 1:
                logging.debug("Creation mode TEST, update UI labels.")
                self.ui.storage_directory.setText(self.test_directory)
                logging.debug("Set storage directory: %s" %
                              (self.test_directory))

    else:
        self.ui.storage_directory.setText("Please select valid path")
        self.ui.save.setDisabled(True)
        logging.warning("Invalid path selected, block saving of pictures.")


def select_top_directory_path(self):
    logging.debug("\nSelect storage directory (top) path procedure started.")
    directory_management.create_directories(self)
    self.ui.save.setDisabled(False)
    if self.creation_mode == 0:
        self.ui.storage_directory.setText(self.train_directory)
        logging.debug("Selected storage directory path: %s" %
                      (self.train_directory))

    if self.creation_mode == 1:
        self.ui.storage_directory.setText(self.test_directory)
        logging.debug("Selected storage directory path: %s" %
                      (self.test_directory))


def get_absolute_file_name(ui, label):
    if ui.creation_mode == 0:
        data_directory = ui.train_directory
    if ui.creation_mode == 1:
        data_directory = ui.test_directory

    label_path = os.path.join(data_directory, label)
    ex_amount_of_images = count_images(label_path) + 1
    filename = f'{label_path}/{ex_amount_of_images}_{label}.png'
    return filename


def count_images(label_path):
    return len(os.listdir(label_path))


def check_for_existing_dataset(self):
    global SELECTION_GO
    if scan_directory_for_preexisting_data(self.top_directory):
        choice = warning_message_box(self)
        return choice
    return SELECTION_GO


def scan_directory_for_preexisting_data(selected_top_directory):
    dirs = os.listdir(selected_top_directory)

    for i in directory_management.GREEK:
        if i in dirs:
            return True
    return False


def confirmation_message_box(self):
    choice = QMessageBox.question(self,
                                  "Confirm",
                                  'Dataset directory will be in %s' %
                                  self.top_directory,
                                  QMessageBox.Yes | QMessageBox.No)

    choice = message_box_decider(choice)
    return choice


def warning_message_box(self):
    choice = QMessageBox.question(self, "Warning",
                                  "It looks like the chosen directory already contains a dataset.\n"
                                  "Are you sure that you want to initialize a new dataset directory in here?",
                                  QMessageBox.Yes | QMessageBox.No)
    choice = message_box_decider(choice)
    return choice


def statistic_message_box(self, statistics):
    formatted_letter_statistics = []
    statistics_total = []
    headline = "Dataset statistics"

    for statistic in statistics:
        formatted_statistic = format_statistic(statistic)
        formatted_letter_statistics.append(formatted_statistic[0])
        statistics_total.append(formatted_statistic[1])

    message_box = QMessageBox()
    message_box.setIcon(QMessageBox.Information)
    message_box.setText("%s\n\n%s - %s\n\n%s \n\n%s - %s\n\n%s" %
                        (headline,
                         "TRAIN",
                         statistics_total[0],
                         formatted_letter_statistics[0],
                         "TEST",
                         statistics_total[1], formatted_letter_statistics[1]))
    message_box.setStyleSheet("QLabel{min-height: 600px;}")
    message_box.exec_()


def format_statistic(statistic):
    formatted_statistic = ""
    statistics_total = statistic['Total']

    for inx, stat in enumerate(statistic):
        if inx == 0:
            continue
        if inx % 2 == 0:
            formatted_statistic = ("%s%s" % (formatted_statistic,
                                             "\n"))

        formatted_statistic = ("%s%s%s%s%s" %
                               (formatted_statistic,
                                stat, "  -  ",
                                statistic[stat],
                                "\t\t"))

    return formatted_statistic, statistics_total


def message_box_decider(choice):
    global SELECTION_GO, SELECTION_STOP
    if choice == QMessageBox.Yes:
        return SELECTION_GO

    else:
        return SELECTION_STOP


def compile_design_file():
    logging.debug("\nCompile design file procedure started.")
    file_path = os.path.dirname(os.path.abspath(__file__))

    ui_path = os.path.join(file_path, "UI")
    os.chdir(ui_path)
    logging.debug("Current working directory was changed to: %s" %
                  (os.getcwd()))
    os.system("./translate.sh")
    logging.debug("Transated '.ui' file.")
    dataset_path = os.path.join(file_path, "../Dataset")

    os.chdir(dataset_path)
    logging.debug("Current working directory was changed to: %s" %
                  (os.getcwd()))


def enable_default_storage_directory():
    logging.debug("Manage the current working directory to enable default storage directory.")
    file_path = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(file_path, "../Dataset")
    os.chdir(dataset_path)
    logging.debug("Current working directory was changed to: %s" %
                  (os.getcwd()))


def check_image_changed(self):
    if self.image != self.blank_image:
        return True
    return False
