from . import constants
from warnings import warn
from pandas import DataFrame
from math import ceil
import numpy as np
from sklearn import tree
from sklearn import datasets
from gower import gower_matrix
from imblearn.over_sampling import SMOTENC, SMOTEN


def load_breast_cancer(train_split=constants.SPLIT, cat=True):
    """
    Load toy dataset crafted from the breast cancer wisconsin dataset.

    :param train_split: proportion of training data
    :param cat: boolean flag to specify if the dataset should contain also categorical features

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
    # if cat, convert the first 5 features from continuous to discrete values
    if cat:
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


def __find_nearest_class(target: np.ndarray, data: np.ndarray, data_class: np.ndarray, val_class: 0, min_size=1,
                         cat_list: list = None):
    """
    Finds the closest instance that satisfies the condition of having at least min_size instances of a class different than val_class.

    :param target: target example
    :param data: data where to find the nearest neighbours
    :param data_class: data labels
    :param val_class: value of the majority label
    :param min_size: desired minimum number of neighbours with a label that is not val_class
    :param cat_list: list of booleans to specify which variables are categorical

    :return: index of the closest instance that satisfies the condition
    """
    d_gow = gower_matrix(target, data, cat_features=cat_list)
    d2index = dict(zip(d_gow[0].tolist(), list(range(data.shape[0]))))
    my_index = [d2index[i] for i in np.sort(d_gow)[0].tolist()]
    local_training_set_y = data_class[my_index]  # sorted y
    # first index that satisfies the condition of having at least min_size instances
    # where y!=y_target
    min_index = np.where(local_training_set_y != val_class)[0][min_size - 1]
    return min_index


def __find_neighbours(target: np.ndarray, data: np.ndarray, cat_list: list = None,
                      n=constants.NEIGHBOURHOOD_SIZE):
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
    # Check on n: if less than 1, it has to be considered as % of training set
    if n < 1:
        n = int(ceil(data.shape[0] * n))
    my_index = [d2index[i] for i in np.sort(d_gow)[0][0:n].tolist()]
    # nk nearest neighbours in the training set according to gower distance
    local_training_set = data[my_index, :]
    return local_training_set


def __random_oversample(x_local, x_instance, cat_list: list = None, size: int = 1, uniform: bool = True,
                        seed: int = constants.SEED):
    """
    Local data augmentation by randomly generate new instances. Non-uniform random oversampling will use sample statistics to generate the new instances.

    :param x_local: local set of examples to use for oversampling
    :param x_instance: target example
    :param cat_list: list of booleans to specify which variables are categorical
    :param size: number of new instances to generate
    :param uniform: specify if the new instances must be drawn from uniform distribution or not
    :param seed: specify random state

    :return: oversampled local data
    """
    x = np.concatenate((x_local, x_instance))
    o = np.zeros((size, x.shape[1]))
    np.random.seed(seed)
    for i in range(x.shape[1]):
        column = x[:, i]
        if cat_list is not None and cat_list[i]:
            if uniform:
                o[:, i] = np.random.randint(low=int(np.min(column)), high=int(np.max(column)) + 1, size=size)
            else:
                vals, counts = np.unique(column, return_counts=True)
                o[:, i] = np.random.choice(a=np.unique(column), size=size, p=counts / sum(counts))
        else:
            if uniform:
                o[:, i] = np.random.uniform(low=np.min(column), high=np.max(column), size=size)
            else:  # if not uniform, we can generate from normal distribution
                mu, sigma = np.mean(column), np.std(column)  # mean and standard deviation
                o[:, i] = np.random.normal(mu, sigma, size=size)
    return o


def __oversample(x_local, x_instance,
                 y_local_pred, y_instance_pred,
                 cat_list: list = None, oversampling: str = None,
                 oversampling_size: int = 100, seed: int = constants.SEED):
    """
    Local data augmentation with SMOTE (Synthetic Minority Oversampling TEchnique).

    :param x_local: local set of examples to use for oversampling
    :param x_instance: target example
    :param y_local_pred: predicted class for the local set of examples
    :param y_instance_pred: predicted class for the target instance
    :param cat_list: list of booleans to specify which variables are categorical
    :param oversampling: type of oversampling. Possible values: smote, uniform, non-uniform, none. Default: None
    :param oversampling_size: number of new instances to be generated. Not used when oversampling_type=smote
    :param seed: specify random state

    :return: oversampled local data
    """
    if oversampling == "smote":
        if cat_list is None:
            smote = SMOTEN(random_state=seed, sampling_strategy='all')
        else:
            smote = SMOTENC(categorical_features=np.where(cat_list)[0].tolist(), random_state=seed,
                            sampling_strategy='all')
        return smote.fit_resample(np.concatenate((x_local, x_instance)),
                                  np.append(y_local_pred, y_instance_pred))[0]  # return X only, we don't need y
    else:
        return __random_oversample(x_local=x_local, x_instance=x_instance,
                                   cat_list=cat_list, size=oversampling_size, uniform=(oversampling == "uniform"),
                                   seed=seed)


def __create_tree(X, y, X_features, max_depth=constants.MAX_DEPTH,
                  min_samples_leaf=constants.MIN_SAMPLES_LEAF):
    """
    Grow a classification tree without pruning.
    Note: why a fixed seed? According to this issue (github.com/scikit-learn/scikit-learn/issues/2386), by default the sklearn implementation for decision tree classifiers is NOT deterministic, even if max_features=n_features and splitter=best, because the implementation will still sample them at random from the list of features even though this means all features will be sampled. Thus, the order in which the features are considered is pseudo-random and a deterministic behaviour between different runs can be achieved only if the random state is fixed a priori.


    :param X: training set
    :param y: class to be predicted
    :param X_features: feature names
    :param max_depth: the maximum depth of the tree. If None, no depth-based pruning is applied.
    :param min_samples_leaf: the minimum number of samples required to be at a leaf node. If int, the value is the minimum number. If float, then ceil(min_samples_leaf*n_samples) is the minimum number of samples for each node.
    :param seed: specify random state

    :return: classification tree
    """
    clf_tree_0 = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=1)
    clf_tree_0.fit(DataFrame(X, columns=X_features), y)
    return clf_tree_0


def __sort_unique_by_freq(array: np.array):
    """
    Return unique values of an array, sorted by descending frequency
    """
    unique_elements, frequency = np.unique(array, return_counts=True)
    sorted_indexes = np.argsort(frequency)[::-1]
    return unique_elements[sorted_indexes]


def __get_suggested_min_n_smote(x_target, x_train, y_local, y_train, cat_list):
    """
    Return the minimum neighborhood size for smote
    """
    k = SMOTEN().k_neighbors + 1  # default k used in SMOTE (+1 to include the target instance)
    y_most_freq = __sort_unique_by_freq(y_local)[0]
    return __find_nearest_class(x_target, x_train, y_train, y_most_freq, k, cat_list) + 1


def run(x_target, y_pred_target, x_train, feature_names, cat_list, predict_fun,
        neighbourhood_size=constants.NEIGHBOURHOOD_SIZE, oversampling=constants.OVERSAMPLING,
        oversampling_size=constants.OVERSAMPLING_SIZE,
        max_depth=constants.MAX_DEPTH, min_samples_leaf=constants.MIN_SAMPLES_LEAF, seed=constants.SEED):
    """
    Run the AraucanaXAI algorithm and plot the calssification tree.

    :param x_target:
    :param y_pred_target:
    :param x_train:
    :param feature_names:
    :param cat_list: list of booleans to specify which variables are categorical
    :param predict_fun: function used to predict the outcomes, i.e. the model we want to explain. Function must have one input only: the data.
    :param neighbourhood_size: specify the number of neighbours to consider
    :param oversampling: type of oversampling. Possible values: smote, uniform, non-uniform, none. Default: None
    :param oversampling_size: number of new instances to be generated. Not used when oversampling_type=smote. Default: 100
    :param max_depth: the maximum depth of the tree. If None, no depth-based pruning is applied. Default: None
    :param min_samples_leaf: the minimum number of samples required to be at a leaf node. If int, the value is the minimum number. If float, then ceil(min_samples_leaf*n_samples) is the minimum number of samples for each node. Default: 1
    :param seed: specify random state

    :returns:
        - tree: classification tree
        - data: resampled data
        - acc: accuracy on resampled data
    """

    # 1) FIND NEIGHBOURS
    local_train = __find_neighbours(target=x_target,
                                    data=x_train,
                                    cat_list=cat_list,
                                    n=neighbourhood_size)
    y_local_train = predict_fun(local_train)
    suggested_min_n = __get_suggested_min_n_smote(x_target=x_target,
                                                  x_train=x_train,
                                                  y_local=y_local_train,
                                                  y_train=predict_fun(x_train),
                                                  cat_list=cat_list)

    # 2) OVERSAMPLING
    os_check_passed = False
    if oversampling is not None:
        if oversampling == "smote" and (neighbourhood_size < suggested_min_n):
            warn(
                "Cannot run SMOTE oversampling due to insufficient neighborhood size. Required minimum neighborhood size: %d . Oversampling skipped." % suggested_min_n)
        elif oversampling in ["smote", "uniform", "non-uniform"]:
            os_check_passed = True
        else:
            warn("Unrecognized oversampling type. Supported methods: smote, uniform, non-uniform.")

    if os_check_passed:
        X_res = __oversample(x_local=local_train,
                             x_instance=x_target,
                             y_local_pred=y_local_train,
                             y_instance_pred=y_pred_target,
                             cat_list=cat_list,
                             oversampling=oversampling,
                             oversampling_size=oversampling_size,
                             seed=seed)
        y_res = predict_fun(X_res)
    else:
        X_res = np.concatenate((local_train, x_target))
        y_res = np.append(y_local_train, y_pred_target)

    # 3) CREATE TREE
    xai_c = __create_tree(X_res, y_res, feature_names,
                          max_depth=max_depth, min_samples_leaf=min_samples_leaf, seed=seed)
    return {'tree': xai_c,
            'data': [X_res, y_res],
            'acc': xai_c.score(X_res, y_res)}
