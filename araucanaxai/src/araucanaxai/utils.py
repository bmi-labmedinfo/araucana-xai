from . import constants
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from gower import gower_matrix
from imblearn.over_sampling import SMOTENC


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
    Local data augmentation with oversampling.

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


def __plot_tree(classif_tree, feature_names, class_names):
    """
    Plot a classification tree.

    :param classif_tree: classification tree
    :param feature_names: feature names
    :param class_names: class names
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    tree.plot_tree(classif_tree, feature_names=feature_names, filled=True, class_names=class_names)
    plt.show()


def run(x_target, y_pred_target, data_train, feature_names, cat_list, predict_fun, y_label=['negative', 'positive'], seed=constants.SEED):
    """
    Run the AraucanaXAI algorithm and plot the calssification tree.

    :param x_target: local set of examples to use for oversampling
    :param y_pred_target: target example
    :param data_train: predicted class for the local set of examples
    :param feature_names: predicted class for the target instance
    :param cat_list: list of booleans to specify which variables are categorical
    :param predict_fun: function used to predict the outcomes, i.e. the model we want to explain. Function must have one input only: the data.
    :param seed: specify random state

    :returns:
        - tree: classification tree
        - data: resampled data
        - acc: accuracy on resampled data
    """
    local_train = __find_neighbours(target=x_target,
                                    data=data_train,
                                    cat_list=cat_list)
    y_local_train = predict_fun(local_train)
    X_res, y_res = __oversample(x_local=local_train,
                                x_instance=x_target,
                                y_local_pred=y_local_train,
                                y_instance_pred=y_pred_target,
                                cat_list=cat_list,
                                seed=seed)
    xai_c = __create_tree(X_res, y_res, feature_names, seed=seed)
    __plot_tree(xai_c, feature_names, y_label)
    return {'tree': xai_c,
            'data': [X_res, y_res],
            'acc': xai_c.score(X_res, y_res)}
