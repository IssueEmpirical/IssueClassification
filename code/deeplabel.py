# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import keras
from baselines.utils import load_data_from_csv, write_to_csv, evaluate, create_logger_handler


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis)
        )
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


def create_model(embedding_length, max_seq_len):
    sequence = tf.keras.layers.Input(shape=(max_seq_len, ), dtype='int32')
    embedded_sequences = tf.keras.layers.Embedding(embedding_length, 100)(sequence)

    (lstm, forward_h, forward_c, backward_h, backward_c) \
        = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True, return_state=True))(embedded_sequences)

    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    context_vector, attention_weights = Attention(10)(lstm, state_h)

    output = tf.keras.layers.Dense(2)(context_vector)

    return tf.keras.Model(inputs=sequence, outputs=output)


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

