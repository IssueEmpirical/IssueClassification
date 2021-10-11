# coding=utf-8
"""
Predict label by combining the prediction of title and the prediction of desc
"""
from scipy.special import softmax
import csv
import numpy as np
from baselines.utils import load_data_from_csv, evaluate, create_logger_handler


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


def process_result(results):
    rs = {}
    for item in results:
        _id = str(item[0])
        probs = item[2].replace('[', '').replace(']', '').replace('\n', ' ').replace(',', '').split(' ')
        probs = np.asarray([float(j) for j in probs if j != ''])
        rs[_id] = {
            'pred': int(item[1]),
            1: probs[1],
            0: probs[0]
        }
    return rs


def ensemble(fold, logger):
    truth = {}
    truth_path = ''
    truth_list = read_csv(truth_path)
    for item in truth_list:
        truth[str(item[0])] = int(item[1])

    desc_path = ''
    desc_rs = read_csv(desc_path)
    desc_rs = process_result(desc_rs)

    title_path = ''
    title_rs = read_csv(title_path)
    title_rs = process_result(title_rs)

    y_pred = []
    y_truth = []
    title_pred = []
    desc_pred = []
    for indx, key in enumerate(truth.keys()):
        if key not in title_rs.keys() or key not in desc_rs.keys():
            continue
        y_truth.append(truth[key])

        prob_1_title = title_rs[key][1]
        prob_1_desc = desc_rs[key][1]
        prob_0_title = title_rs[key][0]
        prob_0_desc = desc_rs[key][0]
        title_pred.append(title_rs[key]["pred"])
        desc_pred.append(desc_rs[key]["pred"])

        prob_1 = (prob_1_title + prob_1_desc) / 2
        prob_0 = (prob_0_title + prob_0_desc) / 2

        pred = 1 if prob_1 >= prob_0 else 0

        y_pred.append(pred)

    evaluate(y_truth, y_pred, logger)


if __name__ == '__main__':
    pass



