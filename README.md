<p align="center">
  <img width="25%" align="center" src="https://github.com/MegaJoctan/MALE5/assets/65341461/5a903238-921d-4f09-8e27-1847d4052af3" alt="logo">
</p>
<h1 align="center">
  M A L E 5
</h1>
<p align="center">
 A python-like Machine Learning Library for MQL5
</p>

<p align="center">
  <a href="https://github.com/MegaJoctan/MALE5/releases" target="_blank">
    <img src="https://img.shields.io/github/v/release/MegaJoctan/MALE5?color=%2334D058&label=Version" alt="Version">
  </a>

  <a href="https://github.com/MegaJoctan/MALE5/stargazers">
    <img src="https://img.shields.io/github/stars/MegaJoctan/MALE5?color=brightgreen&label=Stars" alt="Stars"/>
  </a>

  <a href="https://github.com/MegaJoctan/MALE5/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/MegaJoctan/MALE5?color=blue" alt="License"/>
  </a>

  <a>
    <img src="https://img.shields.io/badge/Platform-Win32%20|%20Linux%20|%20macOS-blue?color=blue" alt="Platform Win32 | Linux | macOS"/>
  </a>

</p>

<p align="center">
  <a href="https://discord.gg/2qgcadfgrx" style="text-decoration:none">
    <img src="https://img.shields.io/badge/Discord-%237289DA?style=flat&logo=discord"/>
  </a>
  <a href="https://t.me/fxalgebra_discussion" style="text-decoration:none">
    <img src="https://img.shields.io/badge/Telegram-%232CA5E0?style=flat&logo=telegram"/>
  </a>
</p>

<p align="center">
English | <a href="README_russian.md">Russian</a> 
</p>

## About the Project

MALE5 is a machine-learning repository for creating trading systems in the c++ like, MQL5 programming language.
It was developed to help build machine learning-based trading robots, effortlessly in the [MetaTrader5](https://www.metatrader5.com/en/automated-trading/metaeditor) platform

**My goal is to make the library**

-   **Python-Like, Simple to use** 
-   **Flexible:** To be usable within a Trading Robot(EA), Custom Indicator, and Scripts.
-   **Resource-efficient:** To make it not consume a lot of resources and memory.

**Disclaimer**
*This project is not for MQL5 programming language beginners. Those with prior experience of machine learning using python might find this project easy to understand*

## Installing 

Download the zip file from the releases section extract the library. Go under MQL5 folder in your MetaEditor, from there paste the MALE5 directory you extracted under the Include folder.

![bandicam 2024-07-19 16-20-07-392](https://github.com/user-attachments/assets/e2829b7e-8fc5-4829-98fb-9277da2d86ba)


## Getting Started with the Library

In this project, machine learning models can be placed into three categories. *See the tables below*.

## 01: Predictive Models

Unlike others which are mostly used for analysis and data processing. These AI models when given an inputs (predictors) in matrices or vectors they provide predictions. Currently Available models include.

| **Model Type**               | **Models**                       |
|------------------------------|----------------------------------|
| **Linear Models**            | - Linear regression              |
|                              | - Logistic regression            |
|                              | - Ridge regression               |
|                              | - Polynomial regression          |
| **Decision Trees**           | - Decision tree                  |
| **Ensemble Models**          | - AdaBoost tree                  |
|                              | - Random forest                  |
| **Support Vector Machine**   | - SVM                            |
| **Neural Networks**          | - Regressor Neural Networks      |
|                              | - Pattern Recognition Neural Networks |
|                              | - Kohonen Maps                   |
| **Naïve Bayes**              | - Naïve Bayes models             |



### Traininng the predictive models

All the predictive models functions for training and deploying them for predictions follow similar patterns.

[![Watch the video](https://github.com/user-attachments/assets/7d68fa9f-0c75-4d37-9c0f-b9926be9c430)](https://www.youtube.com/watch?v=wKk85PZ2ra8&t=1s)


#### For Regression Problems

A regression problem is a type of predictive modeling problem where the goal is to predict a continuous output variable based on one or more input features. In other words, given input data, the model aims to determine a continuous value. For example, predicting the next closing price of a financial instrument.

**How Regression Models Work**

Firstly, we import the necessary libraries we need. In this example, we will be using the Decision Tree Regressor.

```MQL5
#include <MALE5\Decision Tree\tree.mqh>
#include <MALE5\preprocessing.mqh>
#include <MALE5\MatrixExtend.mqh> //helper functions for data manipulations
#include <MALE5\metrics.mqh> //for measuring the performance

StandardizationScaler scaler; //standardization scaler from preprocessing.mqh
CDecisionTreeRegressor *decision_tree; //a decision tree regressor model
```

1. **Data Collection**

Gather a dataset with input features (also called predictors or attributes) and the corresponding target variable (the output).

```MQL5
vector open, high, low, close;     
int data_size = 1000;

//--- Getting the open, high, low, and close values for the past 1000 bars, starting from the recent closed bar of 1
open.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_OPEN, 1, data_size);
high.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_HIGH, 1, data_size);
low.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_LOW, 1, data_size);
close.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_CLOSE, 1, data_size);

matrix X(data_size, 3); //creating the X matrix 

//--- Assigning the open, high, and low price values to the X matrix 
X.Col(open, 0);
X.Col(high, 1);
X.Col(low, 2);

vector y = close; // The target variable is the close price
```

2. **Preprocessing**

This involves cleaning and preprocessing the data, which might involve handling missing values, normalizing the data, and encoding categorical variables if possible in the data. 

```MQL5
//--- We split the data into training and testing samples for training and evaluation
matrix X_train, X_test;
vector y_train, y_test;

double train_size = 0.7; //70% of the data should be used for training, the rest for testing
int random_state = 42; //we put a random state to shuffle the data

MatrixExtend::TrainTestSplitMatrices(X, y, X_train, y_train, X_test, y_test, train_size, random_state); // we split the X and y data into training and testing samples         

//--- Normalizing the independent variables
X_train = scaler.fit_transform(X_train); // fit the scaler on the training data and transform the data
X_test = scaler.transform(X_test); // transform the test data

// Print the processed data for verification
//Print("X_train\n",X_train,"\nX_test\n",X_test,"\ny_train\n",y_train,"\ny_test\n",y_test); 
```

3. **Model Selection**

Choose a regression algorithm. Common algorithms for this type of problem include linear regression, decision trees, support vector regression (SVR), k-nearest neighbors (K-NN), and neural networks.

```MQL5
//--- Model selection
decision_tree = new CDecisionTreeRegressor(2, 5); //a decision tree regressor from DecisionTree class
```

4. **Training**

Use the training data to teach the model how to map input features to the correct continuous output values. During training, the model learns patterns in the data.

```MQL5
//--- Training the model
decision_tree.fit(X_train, y_train); // The training function
```

5. **Evaluation**

Assess the model's performance on a separate test dataset to ensure it can generalize well to new, unseen data. Metrics like R-squared (R²) score, mean absolute error (MAE), and root mean squared error (RMSE) are often used to evaluate the model.

```MQL5
//--- Measuring predictive accuracy 
vector train_predictions = decision_tree.predict(X_train);
printf("Decision Tree training R² score = %.3f ", Metrics::r_squared(y_train, train_predictions));

//--- Evaluating the model on out-of-sample predictions
vector test_predictions = decision_tree.predict(X_test);
printf("Decision Tree out-of-sample R² score = %.3f ", Metrics::r_squared(y_test, test_predictions));
```

6. **Prediction**

Once trained and evaluated, the model can be used to predict the continuous values of new, unseen data points.

```MQL5

void OnTick()
{
   //--- Making predictions live from the market 
   CopyRates(Symbol(), PERIOD_D1, 1, 1, rates); // Get the very recent information from the market
   
   vector x = {rates[0].open, rates[0].high, rates[0].low}; // Assigning data from the recent candle in a similar way to the training data
   
   double predicted_close_price = decision_tree.predict(x);
   
   Comment("Next closing price predicted is = ", predicted_close_price);  
}

```

**Types of Regression**

- **Simple Linear Regression:** Predicting a continuous target variable based on a single input feature.
- **Multiple Linear Regression:** Predicting a continuous target variable based on multiple input features.
- **Polynomial Regression:** Predicting a continuous target variable using polynomial terms of the input features.
- **Non-Linear Regression:** Predicting a continuous target variable using non-linear relationships between input features and the target variable.



#### For Classification Problems

A classification problem is a type of predictive modeling problem where the goal is to predict the category or class that a given data point belongs to. In other words, given input data, the model aims to determine which of the predefined classes the input belongs to. Forexample when trying to predict whether the next candle will be either bullish or bearish.

**How Classification Models Work**

Firstly, we import the necessary libraries we need. In this example we will be using the Decision Tree Classifier.

```MQL5

#include <MALE5\Decision Tree\tree.mqh>
#include <MALE5\preprocessing.mqh>
#include <MALE5\MatrixExtend.mqh> //helper functions for data manipulations
#include <MALE5\metrics.mqh> //fo measuring the performance

StandardizationScaler scaler; //standardization scaler from preprocessing.mqh
CDecisionTreeClassifier *decision_tree; //a decision tree classifier model

```

1. **Data Collection** 

Gather a dataset with input features (also called predictors or attributes) and the corresponding class labels (the output).
> 

```MQL5

     vector open, high, low, close;     
     int data_size = 1000;
     
//--- Getting the open, high, low and close values for the past 1000 bars, starting from the recent closed bar of 1
     
     open.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_OPEN, 1, data_size);
     high.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_HIGH, 1, data_size);
     low.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_LOW, 1, data_size);
     close.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_CLOSE, 1, data_size);
     
     decision_tree = new CDecisionTreeClassifier(2, 5);
     
     matrix X(data_size, 3); //creating the x matrix 
   
//--- Assigning the open, high, and low price values to the x matrix 

     X.Col(open, 0);
     X.Col(high, 1);
     X.Col(low, 2);
     
```

2. **Preprocessing**

This involves clearning and preprocess the data, which might involve analytical techniques such as handling missing values, **normalizing the data**, and encoding categorical variables.

```MQL5

//--- We split the data into training and testing samples for training and evaluation
 
     matrix X_train, X_test;
     vector y_train, y_test;
     
     double train_size = 0.7; //70% of the data should be used for training the rest for testing
     int random_state = 42; //we put a random state to shuffle the data so that a machine learning model understands the patterns and not the order of the dataset, this makes the model durable
      
     MatrixExtend::TrainTestSplitMatrices(X, y, X_train, y_train, X_test, y_test, train_size, random_state); // we split the x and y data into training and testing samples         
     

//--- Normalizing the independent variables
   
     X_train = scaler.fit_transform(X_train); // we fit the scaler on the training data and transform the data alltogether
     X_test = scaler.transform(X_test); // we transform the new data this way
     
     //Print("X_train\n",X_train,"\nX_test\n",X_test,"\ny_train\n",y_train,"\ny_test\n",y_test); 

```

3. **Model Selection** 

Choose a classification algorithm. Common algorithms for this type of problem include; logistic regression, decision trees, support vector machines (SVM), k-nearest neighbors (K-NN), and neural networks.

```MQL5

//--- Model selection
   
     decision_tree = new CDecisionTreeClassifier(2, 5); //a decision tree classifier from DecisionTree class

```

4. **Training** 

Use the training data to teach the model how to map input features to the correct class labels. During training, the model learns patterns in the data.

```MQL5

//--- Training the  model
     
     decision_tree.fit(X_train, y_train); //The training function 
     
```

5. **Evaluation**

Assess the model's performance on a separate test dataset to ensure it can generalize well to new, unseen data. Metrics like accuracy score, precision, recall, and F1 score are often used to evaluate the model.

```MQL5

//--- Measuring predictive accuracy 
   
     vector train_predictions = decision_tree.predict_bin(X_train);
     
     printf("Decision decision_tree training accuracy = %.3f ",Metrics::accuracy_score(y_train, train_predictions));

//--- Evaluating the model on out-of-sample predictions
     
     vector test_predictions = decision_tree.predict_bin(X_test);
     
     printf("Decision decision_tree out-of-sample accuracy = %.3f ",Metrics::accuracy_score(y_test, test_predictions)); 


```

6. **Prediction** 

Once trained and evaluated, the model can be used to predict the class labels of new, unseen data points.

Unlike the `predict` function which is used to obtain predictions in regressors. The predictive functions for the classifiers have different slightly for different name starting with the word predict.

- **predict_bin** - This function can be used to predict the classes in as integer values for classification models. For example: The next candle will be ***bullish*** or ***bearish***
- **predict_proba** - This function predicts the probabilities of a certain classes as double values from 0 for a 0% probability chance to 1 for a 100% probability chance. For example: [0.64, 0.36] probability the next candle will be bullish is 64%, while the probability the next candle will be bearish is 36%

```
void OnTick()
  {
     
//--- Making predictions live from the market 
   
   CopyRates(Symbol(), PERIOD_D1, 1, 1, rates); //Get the very recent information from the market
   
   vector x = {rates[0].open, rates[0].high, rates[0].low}; //Assigning data from the recent candle in a similar way to the training data
   
   double predicted_close_price = decision_tree.predict_bin(x);
   
   Comment("Next closing price predicted is = ",predicted_close_price);  
  }

```

**Types of Classification**

- **Binary Classification:** There are only two classes. For example, classifying emails as spam or not spam.
- **Multiclass Classification:** There are more than two classes. For example, classifying types of animals in an image (e.g., cats, dogs, birds).
- **Multilabel Classification:** Each instance can be assigned multiple classes. For example, tagging a photo with multiple labels like "beach", "sunset", and "vacation".




## 02: Clustering Algorithms

These algorithms are for the special purppose of classifying and grouping data with similar patterns and contents together, they excel in data mining situations

| **Model Type**               | **Models**                       |
|------------------------------|----------------------------------|
| **Nearest Neighbors**        | - KNN nearest neighbors          |
| **K-Means Clustering**       | - k-Means clustering algorithm   |
| **Neural Networks**          | - Kohonen Maps                   |

## 03: Dimensionality Reduction Algorithms

These algorithms are widely used in situations where the dataset is huge and needs adjustmets. they excel at reducing the size of datasets without losing much information. Consider them as the algorithms to zip and shread information in the proper way.

| **Model Type**                         | **Models**                          |
|----------------------------------------|-------------------------------------|
| **Dimensionality Reduction Algorithms**| - Linear Discriminant Analysis (LDA)|
|                                        | - Non Negative Matrix Factorization (NMF) |
|                                        | - Principal Component Analysis (PCA) |
|                                        | - Truncated SVD                     |



## Basic Library functionality & helpers

* [MatrixExtend (MatrixExtend.mqh)](https://github.com/MegaJoctan/MALE5/wiki#matrixextendmatrixextendmqh)
* [Cross Validation Library (cross_validation.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Cross-Validation-Library)
* [Linear Algebra Library (linalg.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Linear-Algebra-Library)
* [Kernels library (kernels.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Kernels-Library)
* [Metrics library (metrics.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Metrics-library)
* [Pre-processing library (preprocessing.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Pre-processing-library)
* [Tensor library (Tensor.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Tensor-Library)



## Opening an issue
You can also post bug reports and feature requests (only) in [GitHub issues](https://github.com/MegaJoctan/MALE5/issues).

## Support the Project
If you find this project helpful, Support us by taking one or more of the actions

[BuyMeCoffee](https://www.buymeacoffee.com/omegajoctan)

[OurProducts](https://www.mql5.com/en/users/omegajoctan/seller)

Register to our recommended broker:

[ICMarkets](https://icmarkets.com/?camp=74639)

## Let's work together
[HIRE ME on MQL5.com by clicking on this link](https://www.mql5.com/en/job/new?prefered=omegajoctan)


