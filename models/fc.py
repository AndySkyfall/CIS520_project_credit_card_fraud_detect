"""
Fully connected network architecture 
CIS 520 project - team XYH
Yuxuan Hong
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import util
import sys


def create_model():
    model = tf.keras.models.Sequential([
                # Adds a densely-connected layer with 64 units to the model:
                tf.keras.layers.Dense(64, activation='relu'), #kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                # Add another:
                tf.keras.layers.Dense(64, activation='relu'),
                # Add a softmax layer with 10 output units:
                tf.keras.layers.Dense(2, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def create_model_more_layers():
    model = tf.keras.models.Sequential([
                # Adds a densely-connected layer with 64 units to the model:
                tf.keras.layers.Dense(512, activation='relu'), #kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.2),
                # Add another:
                tf.keras.layers.Dense(512, activation='relu'),# kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation='relu'),
                # Add a softmax layer with 10 output units:
                tf.keras.layers.Dense(2, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    model = create_model()
    model_layers = create_model_more_layers()
    n = opts.upsamplen if opts.upsamplen is not None else 1
    start = n if opts.upsamplestart is None else 1
    all_conf = []
    all_conf_layers = []
    if start > n:
        print("Upsample start should be larger than end")
        sys.exit()
    for t in np.arange(start, n + 1):
        print("t",t)
        needed = util.needed_n(X, y, t)
        temp_X, temp_y = util.upsample(X, y, needed)
        X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, test_size=0.3, random_state=42)
        X_train, X_test = util.normalize(X_train, X_test)
        train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64, drop_remainder=False).shuffle(buffer_size=10000)
        test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
        model.fit(train_dset, epochs=10)
        model_layers.fit(train_dset, epochs=10)
        conf_mat = np.zeros((2, 2), dtype=int)
        conf_mat_layers = np.zeros((2, 2), dtype=int)
        for d, labels in test_dset:
            predictions = model(d)
            predictions_layers = model_layers(d)
            for i in range(len(d)):
                conf_mat[labels[i]][np.argmax(predictions[i])] += 1
                conf_mat_layers[labels[i]][np.argmax(predictions_layers[i])] += 1
        all_conf.append(conf_mat)
        all_conf_layers.append(conf_mat_layers)
    re_fc, re_fc_layer = list(map(lambda x: util.recall(x), all_conf)), list((map(lambda x: util.recall(x), all_conf_layers)))
    up_range = np.arange(start, n+1)
    plt.plot(up_range, re_fc)
    plt.plot(up_range, re_fc_layer)
    plt.title("2-layer NN vs 3-layer NN")
    plt.legend(["2-layer", "3-layer"])
    plt.xlabel("Upsample rate")
    plt.ylabel("Recall")
    plt.show()

if __name__ == '__main__':
    main()
