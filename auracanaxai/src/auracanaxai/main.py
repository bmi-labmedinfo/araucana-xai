from . import constants
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from gower import gower_matrix
from imblearn.over_sampling import SMOTENC

############
# Questa parte serve a generare degli input per testare il resto.
# Tutto ciò non deve essere incluso nel pacchetto

X_train_normalized: DataFrame = DataFrame({'cont': np.random.random(500),
                                           'CAT1': np.random.randint(0,2,500),
                                           'CAT2': np.random.randint(0,2,500)})  # normalized training set
X_feat_list: list = ['cont', 'CAT1', 'CAT2']  # list of features (names)
X_test_normalized: DataFrame = DataFrame({'cont': np.random.random(100),
                                          'CAT1': np.random.randint(0,2,100),
                                          'CAT2': np.random.randint(0,2,100)})  # normalized test set
y_train: Series = Series(np.random.randint(0,2,500))  # true class training set
y_test: Series = Series(np.random.randint(0,2,100))  # true class test set
y_test_gb: Series = Series(np.random.randint(0,2,100))  # predicted class test set

X_test_normalized = X_test_normalized.to_numpy()
X_train_normalized = X_train_normalized.to_numpy()


def clf_predict(data):
    return np.array([np.random.uniform(0, 1) for i in range(0, data.shape[0])])


sel_thr_gb = .5

#####################################
### da qui in poi ricomincia la parte di pacchetto vera e propria


def find_neighbours(target: np.ndarray, data: np.ndarray, cat_list: list, n: int = constants.NEIGHBOURHOOD_SIZE):
    if target.ndim != data.ndim: raise TypeError("target and data must have the same number of dimensions!")
    # Gower distance matrix
    dgow = gower_matrix(target, data, cat_features=cat_list)
    # Let's select the first k neighbours
    d2index = dict(zip(dgow[0].tolist(), list(range(data.shape[0]))))
    myindex = [d2index[i] for i in np.sort(dgow)[0][0:n].tolist()]
    # nk nearest neighbours in the training set according to gower distance
    local_training_set = data[myindex, :]
    return (local_training_set)


# oversampling

def oversample(x_local, x_instance, y_local_pred, y_test_pred, cat_list: list, seed: int = constants.SEED):
    smote_nc = SMOTENC(categorical_features=np.where(cat_list)[0].tolist(), random_state=seed, sampling_strategy='all')
    return smote_nc.fit_resample(np.concatenate((x_local, x_instance)),
                                 np.append(y_local_pred, y_test_pred))

# classification tree

def create_tree(X, y, X_features):
    clf_tree_0 = tree.DecisionTreeClassifier(random_state=constants.SEED)
    clf_tree_0.fit(DataFrame(X, columns=X_features), y)
    print('Acc', clf_tree_0.score(X, y))
    return clf_tree_0

def plot_tree(classif_tree, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(10, 10))
    tree.plot_tree(classif_tree, feature_names=feature_names, filled=True, class_names=class_names)
    plt.show()


######### pipeline

# Let's select the first k neighbours in the training set
# We will use gower distance (it deals with categorical and numerical feat )
cat = ['CAT1', 'CAT2']
iscat = [x in cat for x in X_feat_list]
print('True class:', y_test.iloc[0], '. Predicted class:', y_test_gb[0])

instance = X_test_normalized[0, :].reshape(1, X_test_normalized.shape[1])
local_training_set = find_neighbours(target=instance,
                                     data=X_train_normalized,
                                     cat_list=iscat)

# we use our predictive model to retrieve the predicted y
ylocal_training = (clf_predict(local_training_set) >= sel_thr_gb).astype(int)

X_resampled, y_resampled = oversample(x_local=local_training_set,
                                      x_instance=instance,
                                      y_local_pred=ylocal_training,
                                      y_test_pred=y_test_gb[0],
                                      cat_list= iscat)

c = create_tree(X_resampled, y_resampled, X_feat_list)
plot_tree(c,X_feat_list,['no-death', 'death'])