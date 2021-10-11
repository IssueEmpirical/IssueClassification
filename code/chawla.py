# coding=utf-8
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from baselines.utils import load_data_from_csv, evaluate, write_to_csv, create_logger_handler

import json
import os
import pickle
import pandas as pd


class Chawla(BaseEstimator, ClassifierMixin):
    def __init__(self, configs, logger):
        self.config = configs
        self.logger = logger

        self.train_ids, self.train_x, self.train_y = load_data_from_csv(self.config["train_data_path"])
        self.test_ids, self.test_x, self.test_y = load_data_from_csv(self.config["test_data_path"])

    def fit_and_eval(self):
        vect = CountVectorizer()
        X = vect.fit_transform(self.train_x)
        df2 = pd.DataFrame(X.toarray())
        df2['classification'] = self.train_y
        true_mask = df2['classification'] == 1
        colsums = df2.sum()
        colsums_true = df2.loc[true_mask].sum()
        colsums_false = df2.loc[~true_mask].sum()
        term_memberships = {}
        for column, value in df2.iteritems():
            if column != 'classification':
                term_memberships[vect.get_feature_names()[column]] = (
                    colsums_false.iat[column] / colsums.iat[column],
                    colsums_true.iat[column] / colsums.iat[column],
                )
        
        results = []
        probs = []
        for row in self.test_x:
            vect = CountVectorizer()
            try:
                vect.fit_transform([row])
            except ValueError:
                results.append(0)
                probs.append([1.0, 1.0])
                continue
        
            sum_non_bug = 1.0
            sum_bug = 1.0
            for feature in vect.get_feature_names():
                try:
                    sum_non_bug *= (1 - term_memberships[feature][0])
                    sum_bug *= (1 - term_memberships[feature][1])
                except KeyError:
                    pass
        
            if (1 - sum_bug) > (1 - sum_non_bug):
                results.append(1)
            else:
                results.append(0)
            probs.append([sum_non_bug, sum_bug])
        
        self.logger.info("the predicted results")
        evaluate(self.test_y, results, self.logger)
        
        # save the results of predictions
        results = [(self.test_ids[i], results[i], probs[i]) for i in range(len(results))]
        write_to_csv(os.path.join(self.config["output_path"], "prediction.csv"), results)


if __name__ == '__main__':
    pass