#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import sys
import os
import numpy as np
import tflearn


def min_max_normalize(train_data):
    return (train_data - np.min(train_data)) / (
        np.max(train_data) - np.min(train_data))


def train_test_split_data(X, y, train_size=0.7):
    pos = round(len(X) * train_size)
    X_train, X_test = X[:pos], X[pos:]
    y_train, y_test = y[:pos], y[pos:]
    return X_train, X_test, y_train, y_test


def create_dataset(raw_data):
    # Y is the page view of the next day
    X, y = [], []
    for i in range(0, len(raw_data) - 1):
        X.append(raw_data[i:i + 1])
        y.append(raw_data[i + 1])
    return X, y


def main():
    if len(sys.argv) < 2:
        print('Usage: simple_lstm.py <TRAIN_FILE>')
        sys.exit(-1)
    else:
        train_file = sys.argv[1]

    if not os.path.exists(train_file):
        print('Not found:', train_file)
        sys.exit(-1)

    # read train data from file
    df = pd.read_csv(
        train_file,
        delimiter=',',
        skiprows=7,
        skipfooter=1,
        header=None,
        engine='python',
        usecols=[1])
    raw_data = df.values
    raw_data = raw_data.astype('float32')
    print('n_raw_data :', df.size)

    # Normalize a train_data
    raw_data = min_max_normalize(raw_data)

    # Create dataset from raw_data
    X, y = create_dataset(raw_data)

    # Split dataset into a train and test
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y, train_size=0.8)
    print('Train size:', len(X_train))
    print('Test size:', len(X_test))

    # Define a LSTM model
    net = tflearn.input_data(shape=[None, 1, 1])
    net = tflearn.lstm(net, n_units=6)
    net = tflearn.fully_connected(net, 1, activation='linear')
    net = tflearn.regression(
        net, optimizer='adam', learning_rate=0.001, loss='mean_square')

    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(X_train, y_train, validation_set=0.1, batch_size=1, n_epoch=150)


if __name__ == '__main__':
    main()
