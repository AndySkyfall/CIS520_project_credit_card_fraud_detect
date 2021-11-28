"""
CIS 520 project - team XYH
Yuxuan Hong
Adaboost algorithm working with multiple threshold
"""

from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import util
import sys
import matplotlib.pyplot as plt

def main():
    opts = util.parse_args()
    X, y = util.data_load(opts.dataset)
    n = opts.upsamplen if opts.upsamplen is not None else 1
    start = n if opts.upsamplestart is None else 1
    if start > n:
        print("Upsample start should be larger than end")
        sys.exit()
    thresh = opts.threshold if opts.threshold is not None and opts.threshold >= 0.40 else None
    for t in np.arange(start, n + 1):
        needed = util.needed_n(X, y, t)
        temp_X, temp_y = util.upsample(X, y, needed)
        X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, test_size=0.3, random_state=42)
        X_train, X_test = util.normalize(X_train, X_test)
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        conf_upsample = []
        if thresh is None:
            predictions = clf.predict(X_test)
            conf_mat = confusion_matrix(y_test, predictions)
            conf_upsample.append(conf_mat)
            print(conf_mat)
        else:
            conf_thresh = []
            for i in np.arange(0.4, thresh + 0.01, 0.005):
                predictions = (clf.predict_proba(X_test)[: ,1] >= i).astype(int)
                conf_mat = confusion_matrix(y_test, predictions)
                conf_thresh.append(conf_mat)
                print(i)
                print(conf_mat)
            util.get_roc_curve(conf_thresh, "Adaboost", "threshold")
            plt.show()
    

if __name__ == '__main__':
    main()
