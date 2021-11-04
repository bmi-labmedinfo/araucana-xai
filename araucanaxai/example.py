from pandas import DataFrame, Series
import numpy as np
import araucanaxai

X_train_normalized: DataFrame = DataFrame({'cont': np.random.random(500),
                                           'CAT1': np.random.randint(0, 2, 500),
                                           'CAT2': np.random.randint(0, 2, 500)})  # normalized training set
X_feat_list: list = ['cont', 'CAT1', 'CAT2']  # list of features (names)
X_test_normalized: DataFrame = DataFrame({'cont': np.random.random(100),
                                          'CAT1': np.random.randint(0, 2, 100),
                                          'CAT2': np.random.randint(0, 2, 100)})  # normalized test set
y_train: Series = Series(np.random.randint(0, 2, 500))  # true class training set
y_test: Series = Series(np.random.randint(0, 2, 100))  # true class test set
y_test_gb: Series = Series(np.random.randint(0, 2, 100))  # predicted class test set

X_test_normalized = X_test_normalized.to_numpy()
X_train_normalized = X_train_normalized.to_numpy()


def clf_predict(data, threshold=0.5):
    raw_prob = np.array([np.random.uniform(0, 1) for i in range(0, data.shape[0])])
    return (raw_prob >= threshold).astype(int)


cat = ['CAT1', 'CAT2']
is_cat = [x in cat for x in X_feat_list]
print('True class:', y_test.iloc[0], '. Predicted class:', y_test_gb[0])

instance = X_test_normalized[0, :].reshape(1, X_test_normalized.shape[1])
instance_pred_y = y_test_gb[0]

tree = auracanaxai.run(x_target=instance, y_pred_target=instance_pred_y,
                       data_train=X_train_normalized, feature_names=X_feat_list, cat_list=is_cat,
                       predict_fun=clf_predict, y_label=['no death', 'death'])
