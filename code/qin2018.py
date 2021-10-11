# coding=utf-8
import tensorflow as tf
import numpy as np
import os
from baselines.utils import load_data_from_csv, write_to_csv, evaluate, create_logger_handler, CountTokenizer


def create_model(embedding_length, max_seq_len):
    # create the LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(embedding_length, 100, input_length=max_seq_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2)
    ])
    return model


def generate_sequence(text, tokenizer, seq_length):
    idx = tokenizer.texts_to_sequences(text)
    idx_padded = tf.keras.preprocessing.sequence.pad_sequences(idx, maxlen=seq_length, padding='post',
                                                               truncating='post')
    return idx_padded


def preprocess_inputs(train_data_path, test_data_path, vocab_size, max_seq_len):
    id_train, X_train, y_train = load_data_from_csv(train_data_path)
    id_test, X_test, y_test = load_data_from_csv(test_data_path)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train)

    X_train = generate_sequence(X_train, tokenizer, max_seq_len)
    X_test = generate_sequence(X_test, tokenizer, max_seq_len)

    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)

    return id_train, X_train, y_train, id_test, X_test, y_test


if __name__ == '__main__':
    pass

