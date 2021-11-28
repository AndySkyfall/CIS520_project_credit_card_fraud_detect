import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import sys
import matplotlib.pyplot as plt

import util
import fc


def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)

    fc_nn_model = fc.create_model()
    ada_model = AdaBoostClassifier(n_estimators=100, random_state=0)
    svm_model = SVC(C=1000,gamma=0.1)
    
    n = opts.upsamplen if opts.upsamplen is not None else 1
    start = n if opts.upsamplestart is None else 1

    if start > n:
        print("upsample range error")
        sys.exit()
    conf_fc, conf_ada, conf_svm = [], [], []
    for t in np.arange(start, n + 1):
        needed = util.needed_n(X, y, t)
        temp_X, temp_y = util.upsample(X, y, needed)
        X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, test_size=0.3, random_state=42)
        X_train, X_test = util.normalize(X_train, X_test)
        train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64, drop_remainder=False).shuffle(buffer_size=10000)
        test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64)
        ada_model.fit(X_train, y_train)
        svm_model.fit(X_train, y_train)
        fc_nn_model.fit(train_dset, epochs=10)
        pred_ada = ada_model.predict(X_test)
        pred_svm = svm_model.predict(X_test)
        conf_ada.append(confusion_matrix(y_test, pred_ada))
        conf_svm.append(confusion_matrix(y_test, pred_svm))
        temp = np.zeros((2, 2), dtype=int)
        for d, labels in test_dset:
            predictions = fc_nn_model(d)
            for i in range(len(d)):
                temp[labels[i]][np.argmax(predictions[i])] += 1
        conf_fc.append(temp)
    recall_fc = list(map(lambda x: util.recall(x), conf_fc))
    recall_ada = list(map(lambda x: util.recall(x), conf_ada))
    recall_svm = list(map(lambda x: util.recall(x), conf_svm))
    up_range = np.arange(start, n+1)
    d = {"SVM": recall_svm, "Adaboost": recall_ada, "FC_NN": recall_fc}
    legends = ["SVM", "Adaboost", "FC_NN"]
    for key in d:
        plt.plot(up_range, d[key])
    plt.title("Recall Vs Upsample Graph")
    plt.xlabel("Upsample rate")
    plt.ylabel("Recall")
    plt.legend(legends)
    plt.show()

if __name__ == '__main__':
    main()
