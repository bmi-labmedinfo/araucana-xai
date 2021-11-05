import araucanaxai
from sklearn.linear_model import LogisticRegression
from sklearn import tree
import matplotlib.pyplot as plt

#load toy dataset
data = araucanaxai.load_breast_cancer()

#specify which features are categorical
cat = data["feature_names"][0:3]
is_cat = [x in cat for x in data["feature_names"]]

#train classifier
classifier = LogisticRegression(random_state=42, solver='liblinear', penalty='l1')
classifier.fit(data["X_train"], data["y_train"])
y_test_pred = classifier.predict(data["X_test"])

#declare the instance we want to explain
instance = data["X_test"][0, :].reshape(1, data["X_test"].shape[1])
instance_pred_y = y_test_pred[0]

#build xai tree to explain the instance classification
xai_tree = araucanaxai.run(x_target=instance, y_pred_target=instance_pred_y,
                       data_train=data["X_train"], feature_names=data["feature_names"], cat_list=is_cat,
                       predict_fun=classifier.predict)

#plot the tree
fig, ax = plt.subplots(figsize=(10, 10))
tree.plot_tree(xai_tree['tree'], feature_names=data["feature_names"], filled=True, class_names=data["target_names"])
plt.show()
# or just save it
# plt.savefig('tree.svg', format='svg', bbox_inches="tight")
