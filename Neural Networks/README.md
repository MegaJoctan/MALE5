## Kohonen Maps (Self-Organizing Maps)

This documentation explains the `CKohonenMaps` class in MQL5, which implements **Kohonen Maps**, also known as **Self-Organizing Maps (SOM)**, for clustering and visualization tasks.

**I. Kohonen Maps Theory:**

Kohonen Maps are a type of **artificial neural network** used for unsupervised learning, specifically for **clustering** and **visualization** of high-dimensional data. They work by:

1. **Initializing a grid of neurons:** Each neuron is associated with a weight vector representing its position in the high-dimensional space.
2. **Iteratively presenting data points:**
    * For each data point:
        * Find the **winning neuron** (closest neuron in terms of distance) based on the weight vectors.
        * Update the weights of the winning neuron and its **neighborhood** towards the data point, with decreasing influence as the distance from the winning neuron increases.
3. **Convergence:** After a certain number of iterations (epochs), the weight vectors of the neurons become organized in a way that reflects the underlying structure of the data.

**II. CKohonenMaps Class:**

The `CKohonenMaps` class provides functionalities for implementing Kohonen Maps in MQL5:

**Public Functions:**

* **CKohonenMaps(uint clusters=2, double alpha=0.01, uint epochs=100, int random_state=42)** Constructor, allows setting hyperparameters:
    * `clusters`: Number of clusters (default: 2).
    * `alpha`: Learning rate (default: 0.01).
    * `epochs`: Number of training epochs (default: 100).
    * `random_state`: Random seed for reproducibility (default: 42).
* `~CKohonenMaps(void)` Destructor.
* `void fit(const matrix &x)` Trains the model on the provided data (`x`).
* `int predict(const vector &x)` Predicts the cluster label for a single data point (`x`).
* `vector predict(const matrix &x)` Predicts cluster labels for multiple data points (`x`).

**Internal Functions:**

* `Euclidean_distance(const vector &v1, const vector &v2)`: Calculates the Euclidean distance between two vectors.
* `CalcTimeElapsed(double seconds)`: Converts seconds to a human-readable format (not relevant for core functionality).

**III. Additional Notes:**

* The class internally uses the `CPlots` class (not documented here) for potential visualization purposes.
* The `c_matrix` and `w_matrix` member variables store the cluster assignments and weight matrix, respectively.
* Choosing the appropriate number of clusters is crucial for the quality of the results.

By understanding the theoretical foundation and functionalities of the `CKohonenMaps` class, MQL5 users can leverage Kohonen Maps for:

* **Clustering:** Group similar data points together based on their features.
* **Data visualization:** Project the high-dimensional data onto a lower-dimensional space (e.g., a 2D grid of neurons) for easier visualization, potentially using the `CPlots` class.


## Pattern Recognition Neural Network 

This documentation explains the `CPatternNets` class in MQL5, which implements a **feed-forward neural network** for pattern recognition tasks.

**I. Neural Network Theory:**

A feed-forward neural network consists of interconnected layers of **neurons**. Each neuron receives input from the previous layer, applies an **activation function** to transform the signal, and outputs the result to the next layer.

**II. CPatternNets Class:**

The `CPatternNets` class provides functionalities for training and using a feed-forward neural network for pattern recognition in MQL5:

**Public Functions:**

* `CPatternNets(matrix &xmatrix, vector &yvector,vector &HL_NODES, activation ActivationFx, bool SoftMaxLyr=false)` Constructor:
    * `xmatrix`: Input data matrix (rows: samples, columns: features).
    * `yvector`: Target labels vector (corresponding labels for each sample in `xmatrix`).
    * `HL_NODES`: Vector specifying the number of neurons in each hidden layer.
    * `ActivationFx`: Activation function (enum specifying the type of activation function).
    * `SoftMaxLyr`: Flag indicating whether to use a SoftMax layer in the output (default: False).
* `~CPatternNets(void)` Destructor.
* `int PatternNetFF(vector &in_vector)` Performs a forward pass on the network with a single input vector and returns the predicted class label.
* `vector PatternNetFF(matrix &xmatrix)` Performs a forward pass on the network for all rows in the input matrix and returns a vector of predicted class labels.

**Internal Functions:**

* `SoftMaxLayerFX(matrix<double> &mat)`: Applies the SoftMax function to a matrix (used for the output layer if `SoftMaxLyr` is True).

**III. Class Functionality:**

1. **Initialization:**
    * The constructor validates data dimensions and parses user-defined hyperparameters.
    * The network architecture (number of layers and neurons) is determined based on the provided configuration.
    * Weights (connections between neurons) and biases (individual offsets for each neuron) are randomly initialized.

2. **Forward Pass:**
    * The provided input vector is fed into the first layer.
    * Each layer performs the following steps:
        * Calculates the weighted sum of the previous layer's outputs.
        * Adds the bias term to the weighted sum.
        * Applies the chosen activation function to the result.
    * This process continues through all layers until the final output layer is reached.

3. **SoftMax Layer (Optional)**
    * If the `SoftMaxLyr` flag is True, the output layer uses the SoftMax function to ensure the output values sum to 1 and represent class probabilities.

4. **Prediction:**
    * For single-sample prediction (`PatternNetFF(vector &in_vector)`), the class label with the **highest output value** is returned.
    * For batch prediction (`PatternNetFF(matrix &xmatrix)`), a vector containing the predicted class label for each sample in the input matrix is returned.

**IV. Additional Notes:**

* The class provides several debug statements (disabled by default) to print intermediate calculations for debugging purposes.
* The code uses helper functions from the `MatrixExtend` class (not documented here) for matrix and vector operations.
* Choosing the appropriate network architecture, activation function, and learning approach (not implemented in this class) is crucial for optimal performance on specific tasks.

By understanding the theoretical foundation and functionalities of the `CPatternNets` class, MQL5 users can leverage neural networks for various pattern recognition tasks, including:

* **Classification:** Classifying data points into predefined categories based on their features.
* **Anomaly detection:** Identifying data points that deviate significantly from the expected patterns.
* **Feature learning:** Extracting hidden patterns or representations from the data.


## Regression Neural Network

This documentation explains the `CRegressorNets` class in MQL5, which implements a **Multi-Layer Perceptron (MLP)** for regression tasks.

**I. Regression vs. Classification:**

* **Regression:** Predicts continuous output values.
* **Classification:** Assigns data points to predefined categories.

**II. MLP Neural Network:**

An MLP is a type of **feed-forward neural network** used for supervised learning tasks like regression. It consists of:

* **Input layer:** Receives the input data.
* **Hidden layers:** Process and transform the information.
* **Output layer:** Produces the final prediction (continuous value in regression).

**III. CRegressorNets Class:**

The `CRegressorNets` class provides functionalities for training and using an MLP for regression in MQL5:

**Public Functions:**

* `CRegressorNets(vector &HL_NODES, activation ActivationFX=AF_RELU_)` Constructor:
    * `HL_NODES`: Vector specifying the number of neurons in each hidden layer.
    * `ActivationFX`: Activation function (enum specifying the type of activation function).
* `~CRegressorNets(void)` Destructor.
* `void fit(matrix &x, vector &y)` Trains the model on the provided data (`x` - features, `y` - target values).
* `double predict(vector &x)` Predicts the output value for a single input vector.
* `vector predict(matrix &x)` Predicts output values for all rows in the input matrix.

**Internal Functions (not directly accessible)**

* `CalcTimeElapsed(double seconds)`: Calculates and returns a string representing the elapsed time in a human-readable format (not relevant for core functionality).
* `RegressorNetsBackProp(matrix& x, vector &y, uint epochs, double alpha, loss LossFx=LOSS_MSE_, optimizer OPTIMIZER=OPTIMIZER_ADAM)`: Performs backpropagation for training (details not provided but likely involve calculating gradients and updating weights and biases).
* Optimizer functions (e.g., `AdamOptimizerW`, `AdamOptimizerB`) Implement specific optimization algorithms like Adam for updating weights and biases during training.

**Other Class Members:**

* `mlp_struct mlp`: Stores information about the network architecture (inputs, hidden layers, and outputs).
* `CTensors*` pointers: Represent tensors holding weights, biases, and other internal calculations (specific implementation likely relies on a custom tensor library).
* `matrix` variables: Used for calculations during training and may hold temporary data (e.g., `W_MATRIX`, `B_MATRIX`).
* `vector` variables: Store network configuration details (e.g., `W_CONFIG`, `HL_CONFIG`).
* `bool isBackProp`: Flag indicating if backpropagation is being performed (private).
* `matrix` variables: Used for storing intermediate results during backpropagation (e.g., `ACTIVATIONS`, `Partial_Derivatives`).

**IV. Additional Notes:**

* The class provides various activation function options and supports different loss functions (e.g., Mean Squared Error, Mean Absolute Error) for selecting the appropriate evaluation metric during training.
* The class implements the Adam optimizer, one of several optimization algorithms used for efficient training of neural networks.
* Detailed implementation of the backpropagation algorithm is not provided but is likely the core functionality for training the network.


**Reference**
* [Data Science and Machine Learning (Part 12): Can Self-Training Neural Networks Help You Outsmart the Stock Market?](https://www.mql5.com/en/articles/12209)
* [Data Science and Machine Learning — Neural Network (Part 01): Feed Forward Neural Network demystified](https://www.mql5.com/en/articles/11275)
* [Data Science and Machine Learning — Neural Network (Part 02): Feed forward NN Architectures Design](https://www.mql5.com/en/articles/11334)
