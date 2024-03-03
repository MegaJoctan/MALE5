## Linear Support Vector Machine (SVM)

This documentation explains the `CLinearSVM` class in MQL5, which implements a **linear support vector machine (SVM)** for binary classification tasks.

**I. SVM Theory:**

SVMs are a type of **supervised learning** algorithm used for classification. They aim to find a **hyperplane** that best separates data points belonging to different classes. 

**II. CLinearSVM Class:**

The `CLinearSVM` class provides functionalities for training and using a linear SVM classifier in MQL5:

**Public Functions:**

* **CLinearSVM(uint batch_size=32, double alpha=0.001, uint epochs= 1000,double lambda=0.1):** Constructor:
    * `batch_size`: Size of mini-batches used during training (default: 32).
    * `alpha`: Learning rate for updating weights (default: 0.001).
    * `epochs`: Number of training epochs (default: 1000).
    * `lambda`: Regularization parameter to control overfitting (default: 0.1).
* **~CLinearSVM(void):** Destructor.
* **void fit(matrix &x, vector &y):** Trains the model on the provided data (`x` - features, `y` - binary target labels).
* **int Predict(vector &x):** Predicts the class label (+1 or -1) for a single input vector.
* **vector Predict(matrix &x):** Predicts class labels (+1 or -1) for all rows in the input matrix.

**Internal Variables:**

* `W`: Vector storing the weight coefficients of the hyperplane.
* `B`: Bias term of the hyperplane.
* `is_fitted_already`: Flag indicating if the model is trained.
* `during_training`: Flag indicating if the training process is ongoing.
* `config`: Structure containing hyperparameters for training (batch size, learning rate, epochs, regularization).

**Private Functions:**

* `hyperplane(vector &x)`: Calculates the dot product of the input vector with the weight vector and adds the bias term, representing the hyperplane equation.

**III. Class Functionality:**

1. **Initialization:**
    * The constructor sets default hyperparameter values and initializes internal flags.

2. **Training (fit function):**
    * Checks if the number of samples in the data (`x`) matches the size of the target label vector (`y`).
    * Initializes the weight vector and bias term to zero.
    * Iterates through training epochs:
        * For each epoch, loops through mini-batches of data:
            * Updates the weight and bias terms using the hinge loss function and gradient descent with **stochastic** updates (updating based on a mini-batch).
    * Sets the `is_fitted_already` flag to True.
    * Prints training progress and performance metrics (loss and accuracy) for each epoch during training (debug mode only).

3. **Prediction:**
    * Checks if the model is trained before making predictions.
    * For a single input vector:
        * Calculates the hyperplane output using the weight vector and bias term.
        * Uses the sign of the output to predict the class label (+1 for positive, -1 for negative).
    * For a matrix of input vectors:
        * Iterates through each row and calls the single-sample prediction function.

**IV. Additional Notes:**

* The class implements a linear SVM, which is only effective for linearly separable data.
* The chosen hyperparameters (learning rate, epochs, etc.) can significantly impact the performance of the model. Experimentation and tuning might be needed for optimal results.
* The class uses mini-batch training for improved efficiency on large datasets.
* During training, the `during_training` flag is set to True to prevent making predictions while the model is being updated.
