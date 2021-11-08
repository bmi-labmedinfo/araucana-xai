<div id="top"></div>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<br />
<div align="center">
  <h2>
    Araucana XAI
  </h2>

  <h3 align="center">Tree-based local explanations of machine learning model predictions</h3>

  <p align="center">
    Repository for the <a href="https://test.pypi.org/project/araucanaxai/">araucanaxai</a> package. Implementation of the pipeline described in <a href="https://arxiv.org/abs/2110.08272">Parimbelli et al., 2021</a>.
    <br />
    <a href="https://github.com/detsutut/AraucanaXAI"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/detsutut/AraucanaXAI/issues">Report Bug</a>
    ·
    <a href="https://github.com/detsutut/AraucanaXAI/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Increasingly complex learning methods such as boosting, bagging and deep learning have made ML models more accurate, but harder to understand and interpret. A tradeoff between performance and intelligibility is often to be faced, especially in high-stakes applications like medicine. This project propose a novel methodological approach for generating explanations of the predictions of a generic ML model, given a specific instance for which the prediction has been made, that can tackle both classification and regression tasks. Advantages of the proposed XAI approach include improved fidelity to the original model, the ability to deal with non-linear decision boundaries, and native support to both classification and regression problems.

### Paper
The araucanaxai package implements the pipeline described in <a href="https://arxiv.org/abs/2110.08272">*Tree-based local explanations of machine learning model predictions - Parimbelli et al., 2021*</a>.

XAI-Healthcare International Workshop presentation:<br/>
[![XAI-Healthcare International Workshop](https://img.youtube.com/vi/N22QYvTZFBk/0.jpg)](https://www.youtube.com/watch?v=N22QYvTZFBk)

**Keywords**: *explainable AI, explanations, local explanation, fidelity, interpretability, transparency, trustworthy AI, black-box, machine learning, feature importance, decision tree, CART, AIM*.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- INSTALLATION -->
## Installation

1. Make sure you have the latest version of pip installed
   ```sh
   pip install --upgrade pip
    ```
2. Install araucanaxai through pip
    ```sh
    pip install -i https://test.pypi.org/simple/ araucanaxai
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

Here's a basic example with a built-in toy dataset that illustrates Araucana XAI common usage. 

First, train a classifier on the data. Araucana XAI is model-agnostic, you only have to provide a function that takes data as input and outputs binary labels.

Then, declare the example whose classification you want to explain.

Finally, run the Araucana XAI and plot the xai tree to explain model's decision as a set of IF-ELSE rules.

```python
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


```


<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Basic implementation
- [x] Parameters tuning
- [ ] Fidelity

See the [open issues](https://github.com/detsutut/AraucanaXAI/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact and References

*   **Repository maintainer**: Tommaso M. Buonocore  [![Gmail][gmail-shield]][gmail-url] [![LinkedIn][linkedin-shield]][linkedin-url]  

*   **Paper Link**: [https://arxiv.org/abs/2110.08272](https://arxiv.org/abs/2110.08272)

*   **Project Link**: [https://github.com/detsutut/AraucanaXAI](https://github.com/detsutut/AraucanaXAI)

*   **Package Link**: [https://test.pypi.org/project/araucanaxai/](https://test.pypi.org/project/araucanaxai/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Authors: E. Parimbelli, G. Nicora, S. Wilk, W. Michalowski, R. Bellazzi

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS -->
[contributors-shield]: https://img.shields.io/github/contributors/detsutut/AraucanaXAI.svg?style=for-the-badge
[contributors-url]: https://github.com/detsutut/AraucanaXAI/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/detsutut/AraucanaXAI.svg?style=for-the-badge
[forks-url]: https://github.com/detsutut/AraucanaXAI/network/members
[stars-shield]: https://img.shields.io/github/stars/detsutut/AraucanaXAI.svg?style=for-the-badge
[stars-url]: https://github.com/detsutut/AraucanaXAI/stargazers
[issues-shield]: https://img.shields.io/github/issues/detsutut/AraucanaXAI.svg?style=for-the-badge
[issues-url]: https://github.com/detsutut/AraucanaXAI/issues
[license-shield]: https://img.shields.io/github/license/detsutut/AraucanaXAI.svg?style=for-the-badge
[license-url]: https://github.com/detsutut/AraucanaXAI/blob/master/LICENSE.txt
[linkedin-shield]: 	https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://linkedin.com/in/tbuonocore
[gmail-shield]: https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white
[gmail-url]: mailto:buonocore.tms@gmail.com
