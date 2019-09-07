import os
import csv
import argparse
import datetime
import logging
import directory_management


TRAIN = 0
TEST = 1


class Statistic():

    def __init__(self, args):
        logging.info("\nCreate Statistic object.\n")
        self.sample_statistic_train = dict()
        self.sample_statistic_train["Total"] = 0
        for letter in directory_management.get_letters():
            self.sample_statistic_train[letter] = 0

        self.sample_statistic_test = dict()
        self.sample_statistic_test["Total"] = 0
        for letter in directory_management.get_letters():
            self.sample_statistic_test[letter] = 0

        self.create_statistic_file(args)

    def create_statistic_file(self, args):
        logging.info("\nCreate statistics file procedure started.")

        if os.path.isfile(args.statistics_filename):
            statistics_file = open(args.statistics_filename, "a")
            logging.debug("Existing statistics file found -- %s" %
                          (args.statistics_filename))
            return

        if not os.path.isfile(args.statistics_filename):
            logging.debug("No existing statistics file found.")

            statistics_file = open(args.statistics_filename, "w")
            statistics_file.write("Timestamp,")
            statistics_file.write("Dataset_path,")

            for inx, letter in enumerate(self.sample_statistic_train):
                if inx == 0:
                    statistics_file.write("%s%s" % ("train_", letter))
                    continue
                statistics_file.write("%s%s" % (",train_", letter))

            for letter in self.sample_statistic_test:
                statistics_file.write("%s%s" % (",test_", letter))

            statistics_file.write("\n")

            logging.debug("Statistics file initiated -- %s" %
                          (args.statistics_filename))

            return

    def update_statistics_file(self, storage_directory_path, args):
        logging.debug("\nUpdate statistics file procedure started.")
        with open(args.statistics_filename, "a") as stats_file_write:
            statistics_file_writer = csv.writer(stats_file_write)

            updated_statistics_file_values = []

            timestamp = str(datetime.datetime.now())
            updated_statistics_file_values.append(timestamp.replace(" ", "_"))
            updated_statistics_file_values.append(storage_directory_path)

            for letter in self.sample_statistic_train:
                updated_statistics_file_values.append(self.sample_statistic_train[letter])
                logging.debug("TRAIN: Value %s from statistics obj written." %
                              (self.sample_statistic_train[letter]))

            for letter in self.sample_statistic_test:
                updated_statistics_file_values.append(self.sample_statistic_test[letter])
                logging.debug("TEST: Value %s from statistics obj written." %
                              (self.sample_statistic_test[letter]))

            statistics_file_writer.writerow(updated_statistics_file_values)

            logging.debug("Update statistics file procedure finished.")

    def get_statistic(self):
        return self.sample_statistic_train, self.sample_statistic_test

    def collect_statistics_from_storage_directory(self, args, train_directory, test_directory):
        self.scan_directory(args, train_directory, 0)
        self.scan_directory(args, test_directory, 1)

    def scan_directory(self, args, directory_path, mode):
        logging.debug("\nScan data storage directory procedure started.")
        total_samples = 0
        total_files = 0
        for dir_path, subdir_list, file_list in os.walk(directory_path):
            total_files = total_files + len(file_list)

            for subdir in subdir_list:
                sample_counter = 0
                subdir_full_path = os.path.join(dir_path, subdir)

                with os.scandir(subdir_full_path) as iterator:
                    for entry in iterator:
                        if entry.is_file():
                            sample_counter += 1
                    total_samples = total_samples + sample_counter

                if mode == TRAIN:
                    self.sample_statistic_train[subdir] = sample_counter
                    logging.debug("TRAIN Found samples %s, quantity = %s" %
                                  (subdir, sample_counter))
                if mode == TEST:
                    self.sample_statistic_test[subdir] = sample_counter
                    logging.debug("TEST Found samples %s, quantity = %s" %
                                  (subdir, sample_counter))

        if mode == TRAIN:
            self.sample_statistic_train['Total'] = total_samples
            logging.debug("Mode = %s; Total files = %s & Total samples = %s" %
                          (mode, total_files, total_samples))

        if mode == TEST:
            self.sample_statistic_test['Total'] = total_samples
            logging.debug("Mode = %s; Total files = %s & Total samples = %s" %
                          (mode, total_files, total_samples))

        logging.debug("Scan data storage directory procedure finished.")


def argparse_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-sf',
                        '--statistics-filename',
                        help='Optional specification of the path of the dataset statistics filename.',
                        default='dataset_statistics.csv')
    parser.add_argument('-d',
                        '--debug',
                        help='Flag to enable verbose debug output.',
                        action='store_true')
    return parser
