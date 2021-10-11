# coding=utf-8
from baselines.utils import evaluate, create_logger_handler
import csv


def read_csv(path):
    """
    Read data from csv file
    :param path: the path of csv file
    :return:
    """
    with open(path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        results = [item for item in csv_reader if len(item) != 0]
    return results


def trivial(fold, logger):
    truth_path = ""
    truth_list = read_csv(truth_path)
    truths = [int(item[1]) for item in truth_list]
    preds = [1 for _ in truth_list]

    evaluate(truths, preds, logger)


if __name__ == '__main__':
    pass