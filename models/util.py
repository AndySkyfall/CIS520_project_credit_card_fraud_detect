import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import optparse
import sys
import matplotlib.pyplot as plt



def parse_args():
    """Parse command line arguments (train and test arff files)."""
    parser = optparse.OptionParser(description='run ensemble method')

    parser.add_option('-d', '--dataset', type='string', help='path to' +\
        ' data file')
    parser.add_option('-r', '--upsamplet', type='float', help='ratio of upsample' +\
            'examples with true label')
    parser.add_option('-n', '--upsamplen', type='float', help='upsample by n times')
    parser.add_option('-t', '--threshold', type='float', help='threshold used by adaboost')
    parser.add_option('-s', '--upsamplestart', type='int', help='start point for upsample range')
    (opts, args) = parser.parse_args()

    mandatories = ['dataset',]
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

# Read in and shuffle data
def data_load(filename):
    data = pd.read_csv(filename).to_numpy(dtype=np.float32)
    X, y = data[:,1 :-1], data[:, -1]
    X, y = shuffle(X, y)
    return X, y

# Data normalization
def normalize(X_train, X_test):
    mean_pixel = X_train.mean(axis=(0, 1), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1), keepdims=True)
    X_train = X_train - mean_pixel
    X_train = X_train / std_pixel
    X_test = X_test - mean_pixel
    X_test = X_test / std_pixel
    return X_train, X_test

# Draw data points randomly with replacement
def upsample(X, y, needed):
    count_true = np.sum(y)
    tx = extract_true(X, y)
    result_X, result_y = X, y
    for i in range(needed):
        idx = int(count_true * np.random.random_sample())
        result_X = np.append(result_X, [tx[idx]], axis=0)
        result_y = np.append(result_y, 1)

    assert(len(result_X) == len(result_y))
    return result_X, result_y

# Take all examples with true label
def extract_true(X, y):
    assert(len(X) == len(y))
    result_x = X[y == 1.0]
    return result_x

# @Deprecated
# No longer in use since upsample process takes too much time
# Helper function for upsampling. Compute how many data needed to make true class 
# take up ratio percent of the final dataset
def needed_total(X, y, ratio):
    count = len(y)
    count_true = np.sum(y)
    needed = int((ratio*count-count_true) / (1-ratio))
    return needed

# Helper function for upsampling. Compute how many data needed to make true class
# n times more
def needed_n(X, y, n):
    needed =int(np.sum(y) * (n-1))
    return needed

def recall(x):
    assert len(x) == 2
    assert len(x[0]) == 2
    return x[1][1] / (x[1][1] + x[1][0])

# Take a number of confusion matrix, generate a roc curve
def get_roc_curve(mats, model_name, param_name):
    xs, ys = [], []
    for confusion_matrix in mats:
        false_positive = confusion_matrix[0][1] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        true_positive = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
        xs.append(false_positive)
        ys.append(true_positive)
    plt.plot(xs, ys, 'b')
    plt.title("Roc curve for Credit Card Fraud Detection using " + model_name + " with different " + param_name)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.show()

# plot recalls vs upsample graph for 3 models
# input: model_cm_dict -> {model_name: confusion matrices}
def plot_recall_upsample_curve(model_cm_dict,upsample_range):
    model_names=[]
    for i in model_cm_dict.items():
        recalls=[]
        model_name=i[0]
        model_cms=i[1]
        model_names.append(model_name)
        for matrix in model_cms:
            recall=matrix[1][1] / (matrix[1][0]+matrix[1][1])
            recalls.append(recall)
        plt.plot(upsample_range,recalls)
    plt.title("Recall Vs Upsample Graph for 3 models")
    plt.xlabel("Upsample")
    plt.ylabel("Recall(TPR)")
    plt.legend([model_names[0],model_names[1],model_names[2]])
    plt.show()


