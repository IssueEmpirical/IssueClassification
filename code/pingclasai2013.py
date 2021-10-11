# coding=utf-8
"""
Reimplementation of "Classifying Bug Reports to Bugs and Other Requests Using Topic Modeling"
"""
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import LdaMulticore
from baselines.utils import load_data_from_csv, write_to_csv, evaluate, create_logger_handler, read_csv

import json
import nltk
import pickle
import os
import numpy as np
import pandas as pd


class Pingclasa2013(object):
    def __init__(self, configs, clf, logger):
        self.config = configs

        self.clf = clf

        self.logger = logger

        self.dictionary = None
        self.lda_model = None
        self.train_ids, self.train_x, self.train_y = None, None, None
        self.test_ids, self.test_x, self.test_y = None, None, None

    def build_lda_model(self):
        dictionary_path = os.path.join(self.config["output_path"], "lda_models/topic_model_dictionary.gensim")
        lda_model_path = os.path.join(self.config["output_path"], "lda_models/topic_model.gensim")
        if os.path.exists(dictionary_path) and os.path.exists(lda_model_path):
            self.logger.info("the pre-trained LDA model is existed, load the pre-trained LDA model")
            self.dictionary = Dictionary.load(dictionary_path)
            self.lda_model = LdaModel.load(lda_model_path)

            return
        if not os.path.exists(os.path.join(self.config["output_path"], "lda_models")):
            os.makedirs(os.path.join(self.config["output_path"], "lda_models"))

        self.logger.info("train dictionary")
        self.dictionary = Dictionary(self.train_x)

        self.logger.info("prepare corpus for LDA model")
        corpus = [self.dictionary.doc2bow(text) for text in self.train_x]
        
        self.logger.info("train LDA model")
        self.lda_model = LdaMulticore(corpus, num_topics=50, id2word=self.dictionary, workers=3)
        

    def prepare_data_for_clf(self):
        # extract topic for training dataset and testing dataset
        train_data_topics_path = os.path.join(self.config["lda_model_path"], "lda_models/train_data_topics.csv")
        test_data_topics_path = os.path.join(self.config["lda_model_path"], "lda_models/test_data_topics.csv")

        if os.path.exists(train_data_topics_path) and os.path.exists(test_data_topics_path):
            print("load from cache")
            train_x, train_y, train_ids = self.extract_topics_with_cache(train_data_topics_path)
            test_x, test_y, test_ids = self.extract_topics_with_cache(test_data_topics_path)

            return (train_x, train_y, train_ids), (test_x, test_y, test_ids)

        # first, build the LDA model
        self.build_lda_model()

        # extract the topics
        self.logger.info("build topic-based features")
        train_data_topics = [self.extract_topics_without_cache(x) for x in self.train_x]
        test_data_topics = [self.extract_topics_without_cache(x) for x in self.test_x]

        # save topic-based features to csv file
        train_data = [(self.train_ids[i], train_data_topics[i], self.train_y[i]) for i in range(len(self.train_ids))]
        test_data = [(self.test_ids[i], test_data_topics[i], self.test_y[i]) for i in range(len(self.test_ids))]

        write_to_csv(train_data_topics_path, train_data)
        write_to_csv(test_data_topics_path, test_data)

        return (np.asarray(train_data_topics), np.asarray(self.train_y), self.train_ids), \
               (np.asarray(test_data_topics), np.asarray(self.test_y), self.test_ids)

    def extract_topics_without_cache(self, tokens):
        # prepare topic features for clf when the preprocessed features are not existed
        doc_bow = self.dictionary.doc2bow(tokens)
        topics = self.lda_model.get_document_topics(doc_bow)  # extract topics

        features = [0] * 50
        for topic in topics:
            features[int(topic[0])] = 1

        return features

    def fit_and_eval(self):
        # the training data and testing data for clf which the features are consist of the existence of topics
        (train_x, train_y, train_ids), (test_x, test_y, test_ids) = self.prepare_data_for_clf()

        self.logger.info("train the classifier")
        self.clf.fit(train_x, train_y)


        y_pred = self.clf.predict(test_x)
        y_prob = self.clf.predict_proba(test_x)
        self.logger.info("the predicted results")
        evaluate(test_y, y_pred, self.logger)
        
        results = [(test_ids[i], y_pred[i], y_prob[i]) for i in range(len(y_pred))]
        write_to_csv(os.path.join(self.config["output_path"], "prediction.csv"), results)

    def fuzzy_logic(self):
        (train_x, train_y, train_ids), (test_x, test_y, test_ids) = self.prepare_data_for_clf()
        df2 = pd.DataFrame(train_x)
        df2['classification'] = train_y
        print(df2)
        true_mask = df2['classification'] == 1
        colsums = df2.sum()
        colsums_true = df2.loc[true_mask].sum()
        colsums_false = df2.loc[~true_mask].sum()
        term_memberships = {}
        for column, value in df2.iteritems():
            if column != 'classification':
                term_memberships[column] = (
                    colsums_false.iat[column] / colsums.iat[column],
                    colsums_true.iat[column] / colsums.iat[column],
                )

        results = []
        probs = []
        for row in test_x:

            sum_non_bug = 1.0
            sum_bug = 1.0
            for feature in row:
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
        evaluate(test_y, results, self.logger)

        # save the results of predictions
        results = [(self.test_ids[i], results[i], probs[i]) for i in range(len(results))]
        write_to_csv(os.path.join(self.config["output_path"], "prediction.csv"), results)


    @staticmethod
    def extract_topics_with_cache(path):
        # prepare topic features for clf when the preprocessed features are existed
        orig_data = read_csv(path)
        x_list = []
        y_list = []
        id_list = []
        for item in orig_data:
            id_list.append(str(item[0]))
            x_list.append(eval(item[1]))
            y_list.append(int(item[2]))
        return np.asarray(x_list), np.asarray(y_list), id_list

    @staticmethod
    def load_data(path):
        ids, inputs, labels = load_data_from_csv(path)

        inputs = [nltk.word_tokenize(x) for x in inputs]

        return ids, inputs, labels


if __name__ == '__main__':
    pass