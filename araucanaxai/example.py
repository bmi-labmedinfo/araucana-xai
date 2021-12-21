import araucanaxai
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import *
import matplotlib.pyplot as plt

# load toy dataset with both categorical and numerical features
cat_data = True
data = araucanaxai.load_breast_cancer(train_split=.75, cat=cat_data)
is_cat = None
if cat_data:
    # specify which features are categorical
    cat = data["feature_names"][0:5]
    is_cat = [x in cat for x in data["feature_names"]]

# train logistic regression classifier
classifier = LogisticRegression(random_state=42, solver='liblinear', penalty='l1', max_iter=500)
classifier.fit(data["X_train"], data["y_train"])
y_test_pred = classifier.predict(data["X_test"])

print('precision: ' + str(precision_score(data["y_test"], y_test_pred)) + ', recall: ' + str(
    recall_score(data["y_test"], y_test_pred)))

# declare the instance we want to explain
index = 65
instance = data["X_test"][index, :].reshape(1, data["X_test"].shape[1])
instance_pred_y = y_test_pred[index]

# build xai tree to explain the instance classification
xai_tree = araucanaxai.run(x_target=instance, y_pred_target=instance_pred_y,
                           x_train=data["X_train"],feature_names=data["feature_names"], cat_list=is_cat,
                           neighbourhood_size=150,
                           oversampling="smote", oversampling_size=100,
                           max_depth=3, min_samples_leaf=1,
                           predict_fun=classifier.predict)

# plot the tree
fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(xai_tree['tree'], feature_names=data["feature_names"], filled=True, class_names=data["target_names"])
plt.tight_layout()
# plt.show()
# or just save it
plt.savefig('tree.svg', format='svg', bbox_inches="tight")
