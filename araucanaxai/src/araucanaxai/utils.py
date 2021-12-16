from . import constants
from warnings import warn
from pandas import DataFrame
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


def __find_neighbours(target: np.ndarray, data: np.ndarray, cat_list: list = None,
                      n: int = constants.NEIGHBOURHOOD_SIZE):
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


def __oversample(x_local, x_instance, y_local_pred, y_instance_pred, cat_list: list = None, seed: int = constants.SEED):
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
    ##### wip
    if cat_list is None:
        smote = SMOTEN(random_state=seed, sampling_strategy='all')
    else:
        smote = SMOTENC(categorical_features=np.where(cat_list)[0].tolist(), random_state=seed, sampling_strategy='all')
    return smote.fit_resample(np.concatenate((x_local, x_instance)),
                              np.append(y_local_pred, y_instance_pred))


def __create_tree(X, y, X_features, max_depth=constants.MAX_DEPTH,
                  min_samples_leaf=constants.MIN_SAMPLES_LEAF, seed=constants.SEED):
    """
    Grow a classification tree without pruning.

    :param X: training set
    :param y: class to be predicted
    :param X_features: feature names
    :param max_depth: the maximum depth of the tree. If None, no depth-based pruning is applied.
    :param min_samples_leaf: the minimum number of samples required to be at a leaf node. If int, the value is the minimum number. If float, then ceil(min_samples_leaf*n_samples) is the minimum number of samples for each node.
    :param seed: specify random state

    :return: classification tree
    """
    clf_tree_0 = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed)
    clf_tree_0.fit(DataFrame(X, columns=X_features), y)
    return clf_tree_0


def run(x_target, y_pred_target, data_train, feature_names, cat_list, predict_fun,
        neighbourhood_size=constants.NEIGHBOURHOOD_SIZE, oversampling=constants.OVERSAMPLING,
        max_depth=constants.MAX_DEPTH, min_samples_leaf=constants.MIN_SAMPLES_LEAF, seed=constants.SEED):
    """
    Run the AraucanaXAI algorithm and plot the calssification tree.

    :param x_target: local set of examples to use for oversampling
    :param y_pred_target: target example
    :param data_train: predicted class for the local set of examples
    :param feature_names: predicted class for the target instance
    :param cat_list: list of booleans to specify which variables are categorical
    :param predict_fun: function used to predict the outcomes, i.e. the model we want to explain. Function must have one input only: the data.
    :param neighbourhood_size: specify the number of neighbours to consider
    :param oversampling: specify if neighborhood oversampling should be used. Default: True
    :param max_depth: the maximum depth of the tree. If None, no depth-based pruning is applied. Default: None
    :param min_samples_leaf: the minimum number of samples required to be at a leaf node. If int, the value is the minimum number. If float, then ceil(min_samples_leaf*n_samples) is the minimum number of samples for each node. Default: 1
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
    if len(np.unique(y_local_train)) < 2: # less than 2 classes = no SMOTE oversampling
        warn('Cannot oversample: local y needs to have more than 1 class to perform SMOTE oversampling. Got 1 class '
             'instead.')
        oversampling = False              # set oversampling to false
    if not oversampling:
        X_res = np.concatenate((local_train, x_target))
        y_res = np.append(y_local_train, y_pred_target)
    else:
        X_res, y_res = __oversample(x_local=local_train,
                                    x_instance=x_target,
                                    y_local_pred=y_local_train,
                                    y_instance_pred=y_pred_target,
                                    cat_list=cat_list,
                                    seed=seed)
    xai_c = __create_tree(X_res, y_res, feature_names,
                          max_depth = max_depth, min_samples_leaf = min_samples_leaf, seed=seed)
    return {'tree': xai_c,
            'data': [X_res, y_res],
            'acc': xai_c.score(X_res, y_res)}


def explanation_similarity_araucanaxai(tree_a, tree_b):
    """
    Compute the similarity of the explanations as the concordance between the features lists in the ordered nodes of
    2 trees

    :returns: d: float
    :param tree_a: sklearn.tree._classes.DecisionTreeClassifier obj
    :param tree_b: sklearn.tree._classes.DecisionTreeClassifier obj
    """
    feat_a = tree_a.tree_.feature.copy()
    feat_b = tree_b.tree_.feature.copy()

    # if different len, trim the longer
    n = min(len(feat_a), len(feat_b))
    feat_a = feat_a[0:n]
    feat_b = feat_b[0:n]
    return sum((feat_a == feat_b).astype(int))/n
