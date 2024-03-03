## Linear Discriminant Analysis (LDA) 

This documentation explains the `CLDA` class in MQL5, which implements **Linear Discriminant Analysis (LDA)** for dimensionality reduction and classification tasks.

**I. LDA Theory:**

LDA is a supervised learning technique that aims to find **linear projections** of the data that **maximize the separation between different classes** while **minimizing variance within each class**. This makes it particularly useful for **classification** problems where the goal is to distinguish between distinct groups.

**II. CLDA Class:**

The `CLDA` class provides functionalities for performing LDA in MQL5:

**Public Functions:**

* **CLDA(uint k=NULL, lda_criterion CRITERION_=CRITERION_SCREE_PLOT, double reg_param =1e-6):** Constructor, allows setting hyperparameters:
    * `k`: Number of components to extract (default: None, determined automatically).
    * `CRITERION_`: Criterion for selecting the best components (default: CRITERION_SCREE_PLOT).
    * `reg_param`: Regularization parameter to prevent overfitting (default: 1e-6).
* `~CLDA(void)`: Destructor.
* `matrix fit_transform(const matrix &x, const vector &y):` Trains the model on the provided data (`x` - independent variables, `y` - class labels) and returns the transformed data.
* `matrix transform(const matrix &x):` Transforms new data (`x`) using the trained model.
* `vector transform(const vector &x):` Transforms a single new data point (`x`) using the trained model.

**Internal Functions:**

* `calculate_variance(vector &eigen_values, double threshold=0.95)`: Calculates the number of components based on explained variance (optional).
* `calculate_kaiser(vector &eigen_values)`: Calculates the number of components based on Kaiser's criterion (optional).
* `extract_components(vector &eigen_values, double threshold=0.95)`: Extracts the selected number of components based on the chosen criterion.

**III. Additional Notes:**

* The `m_criterion` member variable allows choosing the criterion for selecting the number of components:
    * **CRITERION_VARIANCE:** Retains components that explain a specific percentage of variance (set by `threshold`).
    * **CRITERION_KAISER:** Retains components with eigenvalues greater than 1.
    * **CRITERION_SCREE_PLOT:** Analyzes the scree plot to visually determine the number of components with significant eigenvalues.
* The `m_regparam` member variable allows for regularization to prevent overfitting.
* The class internally uses the `CPlots` class (not documented here) for potential visualization purposes (e.g., scree plot).

By understanding the theoretical foundation and functionalities of the `CLDA` class, MQL5 users can leverage LDA for dimensionality reduction and classification tasks, particularly when dealing with class separation in their data.




## Principal Component Analysis (PCA) 

This documentation explains the `CPCA` class in MQL5, which implements **Principal Component Analysis (PCA)** for dimensionality reduction and data visualization tasks.

**I. PCA Theory:**

PCA is an unsupervised learning technique that aims to find a **linear transformation** of the data that captures the most **variance** in a **reduced number of dimensions**. This allows for:

* **Dimensionality reduction:** Reduce the number of features while retaining most of the information in the data.
* **Data visualization:** Project high-dimensional data onto a lower-dimensional space for easier visualization.

**II. CPCA Class:**

The `CPCA` class provides functionalities for performing PCA in MQL5:

**Public Functions:**

* **CPCA(int k=0, criterion CRITERION_=CRITERION_SCREE_PLOT):** Constructor, allows setting hyperparameters:
    * `k`: Number of components to extract (default: 0, determined automatically using the chosen criterion).
    * `CRITERION_`: Criterion for selecting the best components (default: CRITERION_SCREE_PLOT).
* `~CPCA(void)` Destructor.
* `matrix fit_transform(matrix &X)` Trains the model on the provided data (`X`) and returns the transformed data.
* `matrix transform(matrix &X)` Transforms new data (`X`) using the trained model.
* `vector transform(vector &X)` Transforms a single new data point (`X`) using the trained model.
* `bool save(string dir)` Saves the model parameters to a specified directory (`dir`).
* `bool load(string dir)` Loads the model parameters from a specified directory (`dir`).

**Internal Functions:**

* `extract_components(vector &eigen_values, double threshold=0.95)`: Extracts the selected number of components based on the chosen criterion (similar to the `CLDA` class).

**III. Additional Notes:**

* The `m_criterion` member variable allows choosing the criterion for selecting the number of components (same options as in `CLDA`).
* The class internally uses the `CPlots` class (not documented here) for potential visualization purposes.
* Saving and loading functionalities allow for model persistence and reusability.

By understanding the theoretical foundation and functionalities of the `CPCA` class, MQL5 users can leverage PCA for dimensionality reduction, data visualization, and potentially feature extraction tasks within their programs.



## Non-Negative Matrix Factorization (NMF) 

This documentation explains the `CNMF` class in MQL5, which implements **Non-Negative Matrix Factorization (NMF)** for data decomposition and feature extraction tasks.

**I. NMF Theory:**

NMF is a dimensionality reduction technique that decomposes a **non-negative matrix** `V` into two **non-negative matrices** `W` and `H`:

* `V` (shape: `m x n`): The input data matrix, where `m` is the number of data points and `n` is the number of features.
* `W` (shape: `m x k`): The **basis matrix**, where `k` is the number of chosen components and each row represents a **basis vector**.
* `H` (shape: `k x n`): The **coefficient matrix**, where each row corresponds to a basis vector and each element represents the **contribution** of that basis vector to a specific feature in the original data.

By finding a suitable factorization, NMF aims to represent the original data as a **linear combination** of basis vectors while preserving the non-negative nature of the input data. This allows for:

* **Dimensionality reduction:** Reduce the number of features while still capturing essential information.
* **Feature extraction:** Identify underlying factors or patterns in the data through the basis vectors.
* **Data interpretation:** Gain insights into the data by analyzing the non-negative contributions of basis vectors to each feature.

**II. CNMF Class:**

The `CNMF` class provides functionalities for performing NMF :

**Public Functions:**

* **CNMF(uint max_iter=100, double tol=1e-4, int random_state=-1):** Constructor, allows setting hyperparameters:
    * `max_iter`: Maximum number of iterations for the NMF algorithm (default: 100).
    * `tol`: Tolerance for convergence (default: 1e-4).
    * `random_state`: Random seed for initialization (default: -1, uses random seed).
* **~CNMF(void):** Destructor.
* **matrix fit_transform(matrix &X, uint k=2):** Trains the model on the provided data (`X`) and returns the decomposed components (`W` and `H`).
    * `k`: Number of components to extract (default: 2).
* **matrix transform(matrix &X):** Transforms new data (`X`) using the trained model.
* **vector transform(vector &X):** Transforms a single new data point (`X`) using the trained model.
* **uint select_best_components(matrix &X):** Analyzes the input data and suggests an appropriate number of components (implementation details might vary depending on the specific NMF algorithm used).

**III. Additional Notes:**

* The internal implementation details of the NMF algorithm might vary depending on the specific chosen library or technique.
* Choosing the appropriate number of components is crucial for optimal performance and avoiding overfitting. The `select_best_components` function is helpful as a starting point, but further evaluation and domain knowledge might be needed for optimal selection.

By understanding the theoretical foundation and functionalities of the `CNMF` class, MQL5 users can leverage NMF for various tasks, including:

* Dimensionality reduction for data visualization or machine learning algorithms that require lower-dimensional inputs.
* Feature extraction to identify underlying structure or patterns in non-negative data.
* Topic modeling for analyzing text data or other types of document collections.

It's important to note that the specific functionalities and implementation details of the `CNMF` class might vary depending on the chosen MQL5 library or framework. Refer to the specific documentation of the library you are using for the most accurate and up-to-date information.



## Truncated Singular Value Decomposition (Truncated SVD) 

This documentation explains the `CTruncatedSVD` class in MQL5, which implements **Truncated Singular Value Decomposition (Truncated SVD)** for dimensionality reduction and data visualization tasks.

**I. Truncated SVD Theory:**

Truncated SVD is a dimensionality reduction technique based on **Singular Value Decomposition (SVD)**. SVD decomposes a matrix `X` into three matrices:

* `U`: A left singular vectors matrix.
* `Σ`: A diagonal matrix containing the singular values of `X`.
* `V^T`: A right singular vectors matrix (transposed).

Truncated SVD retains only **k** top singular values from `Σ` and their corresponding columns from `U` and `V^T`. This creates a lower-dimensional representation of the original data that captures most of the variance.

**II. CTruncatedSVD Class:**

The `CTruncatedSVD` class provides functionalities for performing Truncated SVD in MQL5:

**Public Functions:**

* `CTruncatedSVD(uint k=0):` Constructor, allows setting the number of components (`k`) to retain (default: 0, determined automatically).
* `~CTruncatedSVD(void):` Destructor.
* `matrix fit_transform(matrix& X):` Trains the model on the provided data (`X`) and returns the transformed data.
* `matrix transform(matrix &X):` Transforms new data (`X`) using the trained model.
* `vector transform(vector &X):` Transforms a single new data point (`X`) using the trained model.
* `ulong _select_n_components(vector &singular_values):` (Internal function, not directly exposed to users) Determines the number of components based on the explained variance ratio (implementation details may vary).

**III. Additional Notes:**

* The class internally uses the `CPlots` class (not documented here) for potential visualization purposes.
* Choosing the appropriate number of components (`k`) is crucial for balancing dimensionality reduction and information preservation. The `_select_n_components` function might use different criteria (e.g., explained variance ratio) for automatic selection, and user discretion might be needed depending on the specific task.

By understanding the theoretical foundation and functionalities of the `CTruncatedSVD` class, MQL5 users can leverage Truncated SVD for:

* **Dimensionality reduction:** Reduce the number of features while retaining most of the information.
* **Data visualization:** Project high-dimensional data onto a lower-dimensional space for easier visualization.
* **Feature extraction:** Identify underlying factors or patterns in the data through the singular vectors (although not explicitly mentioned in the class functionalities).

It's important to note that the specific implementation details of the `CTruncatedSVD` class and the `_select_n_components` function might vary depending on the chosen MQL5 library or framework. Refer to the specific documentation of your chosen library for the most accurate and up-to-date information.


**Reference:**
* [Data Science and Machine Learning (Part 18): The battle of Mastering Market Complexity, Truncated SVD Versus NMF](https://www.mql5.com/en/articles/13968)
* [Data Science and Machine Learning(Part 20) : Algorithmic Trading Insights, A Faceoff Between LDA and PCA in MQL5](https://www.mql5.com/en/articles/14128)
