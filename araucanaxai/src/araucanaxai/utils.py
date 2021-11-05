from . import constants
from warnings import warn
from pandas import DataFrame
import numpy as np
from sklearn import tree
from sklearn import datasets
from gower import gower_matrix
from imblearn.over_sampling import SMOTENC


def load_breast_cancer(train_split=constants.SPLIT):
    """
    Load toy dataset crafted from the breast cancer wisconsin dataset.

    :param train_split: proportion of training data

    :returns:
        - X_train: training set
        - y_train: training target class
        - X_test: test set
        - y_test: test target class
        - feature_names: feature names
        - target_names: class names
    """
    cancer = datasets.load_breast_cancer()
    cancer.data = cancer.data[:, 0:10]
    cancer.feature_names = cancer.feature_names[0:10]
    for i in range(0, 5):
        cancer.data[:, i] = cancer.data[:, i] > np.mean(cancer.data[:, i])
        cancer.data[:, i] = cancer.data[:, i].astype(np.int32)
    cancer.feature_names[0:5] = ['radius', 'texture', 'perimeter', 'area', 'smoothness']
    ind = round(len(cancer.data) * train_split)
    return {
        "X_train": cancer.data[0:ind],
        "y_train": cancer.target[0:ind],
        "X_test": cancer.data[ind:len(cancer.data)],
        "y_test": cancer.target[ind:len(cancer.data)],
        "feature_names": cancer.feature_names,
        "target_names": cancer.target_names
    }


def __find_neighbours(target: np.ndarray, data: np.ndarray, cat_list: list, n: int = constants.NEIGHBOURHOOD_SIZE):
    """
    Finds the n nearest neighbours to the target example according to the Gower distance.

    :param target: target example
    :param data: data where to find the nearest neighbours
    :param cat_list: list of booleans to specify which variables are categorical
    :param n: number of neighbours to find

    :return: nearest n neighbours
    """
    if target.ndim != data.ndim: raise TypeError("target and data must have the same number of dimensions!")
    # Gower distance matrix
    d_gow = gower_matrix(target, data, cat_features=cat_list)
    # Let's select the first k neighbours
    d2index = dict(zip(d_gow[0].tolist(), list(range(data.shape[0]))))
    my_index = [d2index[i] for i in np.sort(d_gow)[0][0:n].tolist()]
    # nk nearest neighbours in the training set according to gower distance
    local_training_set = data[my_index, :]
    return local_training_set


def __oversample(x_local, x_instance, y_local_pred, y_instance_pred, cat_list: list, seed: int = constants.SEED):
    """
    Local data augmentation with SMOTE (Synthetic Minority Oversampling TEchnique).

    :param x_local: local set of examples to use for oversampling
    :param x_instance: target example
    :param y_local_pred: predicted class for the local set of examples
    :param y_instance_pred: predicted class for the target instance
    :param cat_list: list of booleans to specify which variables are categorical
    :param seed: specify random state

    :return: oversampled local data
    """
    smote_nc = SMOTENC(categorical_features=np.where(cat_list)[0].tolist(), random_state=seed, sampling_strategy='all')
    return smote_nc.fit_resample(np.concatenate((x_local, x_instance)),
                                 np.append(y_local_pred, y_instance_pred))


def __create_tree(X, y, X_features, seed=constants.SEED):
    """
    Grow a classification tree without pruning.

    :param X: training set
    :param y: class to be predicted
    :param X_features: feature names
    :param seed: specify random state

    :return: classification tree
    """
    clf_tree_0 = tree.DecisionTreeClassifier(random_state=seed)
    clf_tree_0.fit(DataFrame(X, columns=X_features), y)
    return clf_tree_0


def run(x_target, y_pred_target, data_train, feature_names, cat_list, predict_fun, neighbourhood_size=constants.NEIGHBOURHOOD_SIZE, seed=constants.SEED):
    """
    Run the AraucanaXAI algorithm and plot the calssification tree.

    :param x_target: local set of examples to use for oversampling
    :param y_pred_target: target example
    :param data_train: predicted class for the local set of examples
    :param feature_names: predicted class for the target instance
    :param cat_list: list of booleans to specify which variables are categorical
    :param predict_fun: function used to predict the outcomes, i.e. the model we want to explain. Function must have one input only: the data.
    :param neighbourhood_size: specify the number of neighbours to consider
    :param seed: specify random state

    :returns:
        - tree: classification tree
        - data: resampled data
        - acc: accuracy on resampled data
    """
    local_train = __find_neighbours(target=x_target,
                                    data=data_train,
                                    cat_list=cat_list,
                                    n=neighbourhood_size)
    y_local_train = predict_fun(local_train)
    if len(np.unique(y_local_train)) < 2:
        warn('Cannot oversample: local y needs to have more than 1 class to perform SMOTE oversampling. Got 1 class instead.')
        X_res = np.concatenate((local_train, x_target))
        y_res = np.append(y_local_train, y_pred_target)
    else:
        X_res, y_res = __oversample(x_local=local_train,
                                    x_instance=x_target,
                                    y_local_pred=y_local_train,
                                    y_instance_pred=y_pred_target,
                                    cat_list=cat_list,
                                    seed=seed)
    xai_c = __create_tree(X_res, y_res, feature_names, seed=seed)
    return {'tree': xai_c,
            'data': [X_res, y_res],
            'acc': xai_c.score(X_res, y_res)}
