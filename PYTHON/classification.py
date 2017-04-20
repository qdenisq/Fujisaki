
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
import stat_utils
import utils
import matplotlib.cm as cm
import pandas as pd
from numpy.lib.recfunctions import append_fields
import operator

def make_ellipses(gmm, ax):
    for n, color in  enumerate(cm.rainbow(np.linspace(0, 1, gmm.n_components))):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


directory = r'C:\Users\s3628075\Study\Fujisaki\DataBase\enterface\All/'


def rmfield(a, *fields):
    return a[[name for name in a.dtype.names if name not in fields]]


def GMM_classification(directory):
    params=[]
    for i in range(1):
        params.extend(stat_utlis.get_significant_parameters(directory, verbose=True))
    params = np.unique(params)
    print params

    params = ['T0p0', 'Fb', 'Ap0', 'Aa0', 'Aa1', 'T1a0', 'T2a0','T1a1', 'T2a1']
    data = utils.load_obj(directory+'fuj_params_data.pkl')

    data = pd.DataFrame(data)
    # filter out sample with nan in params
    names = data['name']
    data.drop('name', axis=1, inplace=True)
    nan_fraction = {}
    for c in list(data):
        nan_idx = pd.isnull(data[c])
        print sum(nan_idx)
        nan_fraction[c]=float(sum(nan_idx))/len(data)
    print sorted(nan_fraction.items(), key=operator.itemgetter(1))



    i = [i for i in range(len(data)) if not any(np.isnan(v) for v in data[params][i])]
    # names = data['name'][i]

    for n in data.dtype.names:
        nan_len = len(np.isnan(data[n]))
        data[n][np.isnan(data[n])] = 0.0
        mean = sum(data[n]) / len(data[n] - nan_len)
        data[n][np.isnan(data[n])] = mean
        # filtered[n][np.isnan(filtered[n])] = 0.0
        data[n] = (data[n]-min(data[n]))/(max(data[n]) - min(data[n]))*1
    # data = filtered #  np.array(filtered, dtype=data.dtype)
    # create output column
    labels = ['sa', 'su', 'an', 'fe', 'di', 'ha']

    def get_label(name, labels):
        for i in range(len(labels)):
            if labels[i] in name:
                return i
        return None

    label = np.array([get_label(names[j], labels) for j in range(len(data))])
    print len(label), len(data)
    data_array = data[params].view(np.float32).reshape(data.shape + (-1,))
    print data_array.shape
    # data = append_fields(data, 'label', label)

    iris = datasets.load_iris()

    # Break up the dataset into non-overlapping training (75%) and testing
    # (25%) sets.
    skf = StratifiedKFold(label, n_folds=4, shuffle=True)
    # Only take the first fold.
    train_index, test_index = next(iter(skf))

    X_train = data_array[train_index]
    y_train = label[train_index]
    X_test = data_array[test_index]
    y_test = label[test_index]

    n_classes = len(np.unique(y_train))
    # Try GMMs using different types of covariances.
    classifiers = dict((covar_type, GMM(n_components=n_classes,
                                        covariance_type=covar_type, init_params='wc', n_iter=20))
                       for covar_type in ['tied', 'full'])

    n_classifiers = len(classifiers)

    # plt.figure(figsize=(3 * n_classifiers / 2, 6))
    # plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
    #                     left=.01, right=.99)


    X_train[y_train == 0].mean()

    for index, (name, classifier) in enumerate(classifiers.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                      for i in xrange(n_classes)])

        # Train the other parameters using the EM algorithm.
        classifier.fit(X_train)
        h = plt.subplot(1, n_classifiers, index + 1)
        make_ellipses(classifier, h)

        for n, color in enumerate(cm.rainbow(np.linspace(0, 1, len(labels)))):
            data = X_train[y_train == n]
            plt.scatter(data[:, 0], data[:, 1], 0.8, color=color,
                        label=labels[n])
        # Plot the test data with crosses
        for n, color in enumerate(cm.rainbow(np.linspace(0, 1, len(labels)))):
            data = X_test[y_test == n]
            plt.plot(data[:, 0], data[:, 1], 'x', color=color)

        y_train_pred = classifier.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
                 transform=h.transAxes)

        y_test_pred = classifier.predict(X_test)
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
                 transform=h.transAxes)

        plt.xticks(())
        plt.yticks(())
        plt.title(name)

    plt.legend(loc='lower right', prop=dict(size=12))

    plt.show()
    return

from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def get_label(name, labels):
    for i in range(len(labels)):
        if labels[i] in name:
            return i
    return None


def get_labels(names, labels):
    return np.array([get_label(names[j], labels) for j in range(len(names))])

def ADABoost(directory):
    params = stat_utlis.get_significant_parameters(directory, verbose=True)
    data = utils.load_obj(directory+'fuj_params_data.pkl')
    names = data['name']
    data = rmfield(data,'name')

    i = [i for i in range(len(data)) if not any(np.isnan(v) for v in data[params][i])]

    # names = names[i]
    # data = data[i]
    # filtered = np.array([filtered[i] for i in range(len(filtered)) if not any(np.isnan(v) for v in filtered[params][i])], dtype= filtered.dtype)
    # print len(filtered)
    for n in data.dtype.names:
        nan_len = len(np.isnan(data[n]))
        data[n][np.isnan(data[n])] = 0.0
        mean = sum(data[n])/len(data[n] - nan_len)
        data[n][np.isnan(data[n])] = mean
        # filtered[n] = (filtered[n]-min(filtered[n]))/(max(filtered[n]) - min(filtered[n]))*1
    # data = filtered #  np.array(filtered, dtype=data.dtype)
    # create output column
    labels = ['sa', 'su', 'an', 'fe', 'di', 'ha']
    label = get_labels(names, labels)

    features = data[params].view(np.float32).reshape(data.shape + (-1,))
    print len(features)

    skf = StratifiedKFold(label, n_folds=4, shuffle=True)
    # Only take the first fold.
    train_index, test_index = next(iter(skf))

    X_train = features[train_index]
    y_train = label[train_index]
    X_test = features[test_index]
    y_test = label[test_index]

    bdt_real = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        learning_rate=1)

    bdt_discrete = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        learning_rate=1.5,
        algorithm="SAMME")

    bdt_real.fit(X_train, y_train)
    bdt_discrete.fit(X_train, y_train)

    real_test_errors = []
    discrete_test_errors = []

    for real_test_predict, discrete_train_predict in zip(
            bdt_real.staged_predict(X_test), bdt_discrete.staged_predict(X_test)):
        real_test_errors.append(
            1. - accuracy_score(real_test_predict, y_test))
        discrete_test_errors.append(
            1. - accuracy_score(discrete_train_predict, y_test))

    n_trees_discrete = len(bdt_discrete)
    n_trees_real = len(bdt_real)

    # Boosting might terminate early, but the following arrays are always
    # n_estimators long. We crop them to the actual number of trees here:
    discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
    real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
    discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(range(1, n_trees_discrete + 1),
             discrete_test_errors, c='black', label='SAMME')
    plt.plot(range(1, n_trees_real + 1),
             real_test_errors, c='black',
             linestyle='dashed', label='SAMME.R')
    plt.legend()
    plt.ylabel('Test Error')
    plt.xlabel('Number of Trees')

    plt.subplot(132)
    plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
             "b", label='SAMME', alpha=.5)
    plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
             "r", label='SAMME.R', alpha=.5)
    plt.legend()
    plt.ylabel('Error')
    plt.xlabel('Number of Trees')
    plt.ylim((.2,
             max(real_estimator_errors.max(),
                 discrete_estimator_errors.max()) * 1.2))
    plt.xlim((-20, len(bdt_discrete) + 20))

    plt.subplot(133)
    plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
             "b", label='SAMME')
    plt.legend()
    plt.ylabel('Weight')
    plt.xlabel('Number of Trees')
    plt.ylim((0, discrete_estimator_weights.max() * 1.2))
    plt.xlim((-20, n_trees_discrete + 20))

    # prevent overlapping y-axis labels
    plt.subplots_adjust(wspace=0.25)
    plt.show()
    return

GMM_classification(directory)