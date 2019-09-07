import os
import logging


GREEK = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta",
         "Theta", "Iota", "Kappa", "Lambda", "My", "Ny", "Xi", "Omikron",
         "Pi", "Rho", "Sigma", "Tau", "Ypsilon", "Phi", "Chi", "Psi", "Omega"]


def create_directories(ui):
    logging.debug("\nStarted create directories procedure.")
    if not os.path.isdir(ui.top_directory):
        logging.debug("Selected storage directory %s not found." %
                      (ui.top_directory))
        os.makedirs(ui.top_directory)
        logging.debug("Created new top storage directory at: %s" %
                      (ui.top_directory))

    if not os.path.isdir(ui.train_directory):
        logging.debug("Selected train directory %s not found." %
                      (ui.train_directory))
        os.makedirs(ui.train_directory)
        logging.debug("Created new train directory at: %s" %
                      (ui.train_directory))

    if not os.path.isdir(ui.test_directory):
        logging.debug("Selected test directory %s not found." %
                      (ui.test_directory))
        os.makedirs(ui.test_directory)
        logging.debug("Created new test directory at: %s" %
                      (ui.test_directory))

    for i in GREEK:
        train_data_label_path = os.path.join(ui.train_directory, i)
        test_data_label_path = os.path.join(ui.test_directory, i)

        if not os.path.isdir(train_data_label_path):
            os.makedirs(train_data_label_path)
            logging.debug("Train directory created: %s" %
                          (train_data_label_path))
        if not os.path.isdir(test_data_label_path):
            os.makedirs(test_data_label_path)
            logging.debug("Test directory created: %s" %
                          (test_data_label_path))

    logging.debug("Finished create directories procedure.")


def get_letters():
    return GREEK
