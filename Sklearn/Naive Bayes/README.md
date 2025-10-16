## Naive Bayes Classifier

This documentation explains the `CNaiveBayes` class in MQL5, which implements a **Naive Bayes classifier** for classification tasks.

**I. Naive Bayes Theory:**

Naive Bayes is a probabilistic classifier based on **Bayes' theorem**. It assumes that the features used for classification are **independent** of each other given the class label. This simplifies the calculations involved in making predictions.

**II. CNaiveBayes Class:**

The `CNaiveBayes` class provides functionalities for training and using a Naive Bayes classifier in MQL5:

**Public Functions:**

* **CNaiveBayes(void):** Constructor.
* **~CNaiveBayes(void):** Destructor.
* **void fit(matrix &x, vector &y):** Trains the model on the provided data (`x` - features, `y` - target labels).
* **int predict(vector &x):** Predicts the class label for a single input vector.
* **vector predict(matrix &x):** Predicts class labels for all rows in the input matrix.

**Internal Variables:**

* `n_features`: Number of features in the data.
* `y_target`: Vector of target labels used during training.
* `classes`: Vector containing the available class labels.
* `class_proba`: Vector storing the prior probability of each class.
* `features_proba`: Matrix storing the conditional probability of each feature value given each class.
* `c_prior_proba`: Vector storing the calculated prior probability of each class after training.
* `c_evidence`: Vector storing the calculated class evidence for a new data point.
* `calcProba(vector &v_features)`: Internal function (not directly accessible) that likely calculates the class probabilities for a given feature vector.

**III. Class Functionality:**

1. **Training:**
    * The `fit` function takes the input data (features and labels) and performs the following:
        * Calculates the prior probability of each class (number of samples belonging to each class divided by the total number of samples).
        * Estimates the conditional probability of each feature value given each class (using techniques like Laplace smoothing to handle unseen features).
    * These probabilities are stored in the internal variables for later use in prediction.

2. **Prediction:**
    * The `predict` functions take a new data point (feature vector) and:
        * Calculate the class evidence for each class using Bayes' theorem, considering the prior probabilities and conditional probabilities of the features.
        * The class with the **highest class evidence** is predicted as the most likely class for the new data point.

**IV. Additional Notes:**

* The class assumes the data is already preprocessed and ready for use.

**Reference**
* [Data Science and Machine Learning (Part 11): Naïve Bayes, Probability theory in Trading](https://www.mql5.com/en/articles/12184)