# coding=utf-8
import pandas as pd
import csv
import logging
from sklearn import metrics
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from nltk import PorterStemmer


def load_data_from_csv(data_path, is_shuffle=False):
    # load data from csv
    csv_file = pd.read_csv(data_path, names=['id', 'class', 'content'])
    csv_file.content = csv_file.content.astype(str)
    if is_shuffle:
        shuffle_csv = csv_file.sample(frac=1)
    else:
        shuffle_csv = csv_file
    ids = pd.Series(shuffle_csv['id'])
    x = pd.Series(shuffle_csv['content'])
    y = pd.Series(shuffle_csv['class'])
    return list(ids), list(x), list(y)


def write_to_csv(path, data):
    """
    Write data to csv file
    :param path: the saved path of csv file
    :param data: the data need to be wrote
    :return:
    """
    print("the data size is {}".format(len(data)))
    with open(path, 'w', encoding='utf-8', newline='') as csv_ile:
        csv_writer = csv.writer(csv_ile)
        csv_writer.writerows(data)


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


def evaluate(actual, pred, logger):

    m_precision = metrics.precision_score(actual, pred,pos_label=1, average='binary')
    m_recall = metrics.recall_score(actual, pred,pos_label=1, average='binary')
    f1score = metrics.f1_score(actual, pred,pos_label=1, average='binary')
    logger.info('bug precision:{0:.3f}'.format(m_precision))
    logger.info('bug recall:{0:0.3f}'.format(m_recall))
    logger.info('bug f1_score:{0:0.3f}'.format(f1score))

    m_precision = metrics.precision_score(actual, pred, pos_label=0, average='binary')
    m_recall = metrics.recall_score(actual, pred, pos_label=0, average='binary')
    f1score = metrics.f1_score(actual, pred, pos_label=0, average='binary')
    logger.info('nonbug precision:{0:.3f}'.format(m_precision))
    logger.info('nonbug recall:{0:0.3f}'.format(m_recall))
    logger.info('nonbug f1_score:{0:0.3f}'.format(f1score))

    m_precision = metrics.precision_score(actual, pred, average='weighted')
    m_recall = metrics.recall_score(actual,pred, average='weighted')
    f1score = metrics.f1_score(actual, pred, average='weighted')
    logger.info('weighted precision:{0:.3f}'.format(m_precision))
    logger.info('weighted recall:{0:0.3f}'.format(m_recall))
    logger.info('weighted f1_score:{0:0.3f}'.format(f1score))

    logger.info('------For Pingclasai et al.----------')
    m_precision = metrics.precision_score(actual, pred, average='micro')
    m_recall = metrics.recall_score(actual, pred, average='micro')
    f1score = metrics.f1_score(actual, pred, average='micro')
    logger.info('micro precision:{0:.3f}'.format(m_precision))
    logger.info('micro recall:{0:0.3f}'.format(m_recall))
    logger.info('micro f1_score:{0:0.3f}'.format(f1score))


def create_logger_handler(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class CountTokenizer(CountVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(tokenize(doc))


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def porter_stemming(tokens):
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in tokens]
    return stemmed