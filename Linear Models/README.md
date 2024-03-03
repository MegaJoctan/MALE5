## Theory Overview: Linear Regression

Linear regression is a statistical method for modeling the relationship between a dependent variable (what you want to predict) and one or more independent variables (what you are using to predict). It assumes a linear relationship between the variables, meaning the change in the dependent variable is proportional to the change in the independent variable(s).

The goal of linear regression is to find a line (for single independent variable) or hyperplane (for multiple independent variables) that best fits the data points. This line/hyperplane is represented by the following equation:

`y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

where:

* `y` is the dependent variable
* `x₁`, `x₂`, ..., `xₙ` are the independent variables
* `β₀` is the intercept (the y-value where the line/hyperplane crosses the y-axis)
* `β₁`, `β₂`, ..., `βₙ` are the coefficients (slopes) of the independent variables

By fitting the line/hyperplane to the data, linear regression allows us to:

* **Make predictions:** We can use the fitted model to predict the dependent variable for new data points based on their independent variable values.
* **Understand relationships:** The coefficients represent the strength and direction of the relationships between the independent and dependent variables.

## CLinearRegression Class Documentation

The `CLinearRegression` class implements linear regression functionality. It provides methods for both training and prediction:

**Public Functions:**

* `CLinearRegression(void)` Default constructor.
* `~CLinearRegression(void)` Destructor.
* `void fit_LeastSquare(matrix &x, vector &y)` Fits the model using the least squares method. 
    * This method trains the model by finding the coefficients (Betas) that minimize the sum of squared errors between the predicted and actual values.
    * Requires `x` (matrix of independent variables) and `y` (vector of dependent variables) as input.
* `void fit_GradDescent(matrix &x, vector &y, double alpha, uint epochs = 1000)` Fits the model using gradient descent.
    * This method trains the model iteratively, updating the coefficients based on the calculated gradients (slopes of the error function).
    * Requires `x` (matrix of independent variables), `y` (vector of dependent variables), `alpha` (learning rate), and optional `epochs` (number of iterations) as input.
* `double predict(vector &x)` Predicts the dependent variable for a new data point represented by the input vector `x`.
    * Requires a vector containing the values for the independent variables of the new data point.
    * Assumes the model is already trained (trained flag checked internally).
* `vector predict(matrix &x)` Predicts the dependent variables for multiple new data points represented by the input matrix `x`.
    * Requires a matrix where each row represents a new data point with its independent variable values.
    * Assumes the model is already trained (trained flag checked internally).

**Additional Notes:**

* The class uses internal member variables to store the trained coefficients (`Betas` and `Betas_v`).
* Internal helper functions (`checkIsTrained`, `TrimNumber`, `dx_wrt_bo`, `dx_wrt_b1`) are not directly accessible from the user but support the core functionalities.
* The class checks if the model is trained before allowing predictions using the `checkIsTrained` function.



## CLogisticRegression Class: Logistic Regression

The `CLogisticRegression` class provides functionalities for implementing logistic regression in MQL5. This statistical method allows you to model the probability of a binary outcome (belonging to one of two classes) based on independent variables.

**Public Functions:**

* `CLogisticRegression(uint epochs=10, double alpha=0.01, double tol=1e-8)` Constructor, allows setting hyperparameters (epochs, learning rate, tolerance) for training.
* `~CLogisticRegression(void)` Destructor.
* `void fit(matrix &x, vector &y)` Trains the model using the provided training data (`x` - independent variables, `y` - binary classification labels).
    * Internally performs gradient descent optimization to find the optimal weights and bias for the model.
* `int predict(vector &x)` Predicts the class label (0 or 1) for a new data point represented by the input vector `x`.
    * Assumes the model is already trained (checked internally).
* `vector predict(matrix &x)` Predicts class labels for multiple new data points represented by the input matrix `x`.
    * Each row in the matrix represents a new data point.
    * Assumes the model is already trained (checked internally).
* `double predict_proba(vector &x)` Predicts the probability of belonging to class 1 for a new data point represented by the input vector `x`.
    * Assumes the model is already trained (checked internally).
* `vector predict_proba(matrix &x)` Predicts the probabilities of belonging to class 1 for multiple new data points represented by the input matrix `x`.
    * Each row in the matrix represents a new data point.
    * Assumes the model is already trained (checked internally).

**Mathematical Theory (Basic Overview):**

Logistic regression uses the sigmoid function to map the linear combination of the input features (weighted sum) to a probability between 0 and 1. 

The mathematical formula for the logistic function is:

```
f(z) = 1 / (1 + exp(-z))
```

where:

* `z` is the linear combination of weights and features (z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
* `w_i` are the weights for each feature
* `b` is the bias term
* `f(z)` is the predicted probability of belonging to class 1

By adjusting the weights and bias through training (minimizing the error between predicted and actual labels), the logistic regression model learns to distinguish between the two classes based on the input features and predict probabilities accordingly.

**Additional Notes:**

* The class utilizes gradient descent to optimize the weights and bias during training.
* Hyperparameter tuning (epochs, learning rate, tolerance) can significantly impact model performance and should be considered based on the specific data and task.



## CPolynomialRegression Class: Polynomial Regression

The `CPolynomialRegression` class provides functionalities for implementing polynomial regression in MQL5. This technique extends linear regression by fitting a higher-degree polynomial function to the data, allowing for more complex relationships between the independent and dependent variables.

**Public Functions:**

* `CPolynomialRegression(int degree=2)` Constructor, allows setting the degree of the polynomial (default is 2).
* `~CPolynomialRegression(void)` Destructor.
* `void BIC(ulong k, vector &bic, int &best_degree)` Calculates the Bayesian Information Criterion (BIC) for different polynomial degrees (`k`) and recommends the "best" degree based on the lowest BIC value (stored in `best_degree`).
    * Requires `k` (vector of degree values to evaluate), `bic` (output vector to store BIC values), and `best_degree` (output variable to store the recommended degree).
* `void fit(matrix &x, vector &y)` Trains the model using the provided training data (`x` - independent variables, `y` - dependent variables).
    * Internally fits the polynomial function to the data and stores the coefficients in the `Betas` and `Betas_v` member variables.
* `double predict(vector &x)` Predicts the dependent variable for a new data point represented by the input vector `x`.
    * Assumes the model is already trained (trained flag not explicitly mentioned but potentially implemented internally).
* `vector predict(matrix &x)` Predicts the dependent variables for multiple new data points represented by the input matrix `x`.
    * Each row in the matrix represents a new data point.
    * Assumes the model is already trained (trained flag not explicitly mentioned but potentially implemented internally).

**Mathematical Theory (Basic Overview):**

Polynomial regression models the relationship between the independent and dependent variables using a polynomial function of the form:

```
y = β₀ + β₁x + β₂x² + ... + βₙxⁿ
```

where:

* `y` is the dependent variable
* `x` is the independent variable
* `β₀`, `β₁`, ..., `βₙ` are the coefficients of the polynomial (model parameters)
* `n` is the degree of the polynomial (set during object creation or through the `BIC` function)

By increasing the degree of the polynomial, the model can capture more complex non-linear relationships in the data. However, it is crucial to find a balance between model complexity and overfitting (memorizing the training data poorly generalizing to unseen data).

**Additional Notes:**

* The `BIC` function can be used to help select an appropriate polynomial degree by balancing model complexity and goodness-of-fit.
* Choosing a high polynomial degree without sufficient data can lead to overfitting, so careful consideration and potentially additional techniques like regularization might be necessary.


**References**
* [Data Science and Machine Learning (Part 01): Linear Regression](https://www.mql5.com/en/articles/10459)
* [Data Science and Machine Learning (Part 02): Logistic Regression](https://www.mql5.com/en/articles/10626)
* [Data Science and Machine Learning (Part 07): Polynomial Regression](https://www.mql5.com/en/articles/11477)
* [Data Science and Machine Learning (Part 10): Ridge Regression ](https://www.mql5.com/en/articles/11735)
