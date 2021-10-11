# coding=utf-8
"""
Reimplementation of "Automated Classification of Software Bug Reports"
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from baselines.utils import CountTokenizer, DenseTransformer, porter_stemming, load_data_from_csv, write_to_csv, \
    evaluate, create_logger_handler

import json
import os
import pickle

KEYWORDS = ['enhancement', 'refactoring', 'improvement', 'performance', 'feature', 'usability', 'efficiency',
            'contribution', 'reduction', 'support', 'update', 'modification', 'add', 'change', 'configuration']


class Otoom2019(BaseEstimator, ClassifierMixin):
    def __init__(self, configs, clf, logger):
        self.config = configs

        self.clf = clf
        self.logger = logger

        self.train_ids, self.train_x, self.train_y = load_data_from_csv(self.config["train_data_path"])
        self.test_ids, self.test_x, self.test_y = load_data_from_csv(self.config["test_data_path"])

    def fit_and_eval(self):
        text_clf = Pipeline([
            ('vect', CountTokenizer(vocabulary=porter_stemming(KEYWORDS))),
            ('to_dense', DenseTransformer()),
            ('clf', self.clf)
        ])

        self.logger.info("train the classifier")
        text_clf.fit(self.train_x, self.train_y)

        y_pred = text_clf.predict(self.test_x)
        y_prob = text_clf.predict_proba(self.test_x)
        self.logger.info("the predicted results")
        evaluate(self.test_y, y_pred, self.logger)

        results = [(self.test_ids[i], y_pred[i], y_prob[i]) for i in range(len(y_pred))]
        write_to_csv(os.path.join(self.config["output_path"], "prediction.csv"), results)


if __name__ == '__main__':
    pass