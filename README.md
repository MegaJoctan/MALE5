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

**This Library is:**

-   **Simple to use:** You can literally start building your system once you call the class constructor
-   **Flexible:** You can use it in any program script, Indicator, EA's
-   **Resource-efficient:** It doesn't consume a lot of memory or CPU, and takes short time intervals to train

> All the algorithms in this repository works best when the data is [normalized](https://github.com/MegaJoctan/MALE5/wiki/Pre-processing-library#preprocessingmqh-mql5-normalization-techniques), This is crucial for all Machine Learning techniques

## Read the Docs
**Foundation Libraries | modules**
* [MatrixExtend (MatrixExtend.mqh)](https://github.com/MegaJoctan/MALE5/wiki#matrixextendmatrixextendmqh)
* [Cross Validation Library (cross_validation.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Cross-Validation-Library)
* [Linear Algebra Library (linalg.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Linear-Algebra-Library)
* [Kernels library (kernels.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Kernels-Library)
* [Metrics library (metrics.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Metrics-library)
* [Pre-processing library (preprocessing.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Pre-processing-library)
* [Tensor library (Tensor.mqh)](https://github.com/MegaJoctan/MALE5/wiki/Tensor-Library)

**Linear Models**
* [Linear Regression (Linear Regression.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Linear%20Models#theory-overview-linear-regression)
* [Logistic Regression (Logistic Regression.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Linear%20Models#clogisticregression-class-logistic-regression)
* Ridge Regression
* [Polynomial Regression (Polynomial Regression.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Linear%20Models#cpolynomialregression-class-polynomial-regression)

* **Decision Trees**
* [Decision Tree (tree.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Decision%20Tree#decision-trees-in-mql5-classification-and-regression)

**Clustering Techniques**
* DBSCAN
* Hierachical Clustering
* KMeans

**Dimension Reduction Techniques**
* [Linear Discriminant Analysis (LDA.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Dimensionality%20Reduction#linear-discriminant-analysis-lda)
* [Principal Component Analysis (PCA.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Dimensionality%20Reduction#principal-component-analysis-pca)
* [Non-Negative Matrix Factorization (NMF.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Dimensionality%20Reduction#non-negative-matrix-factorization-nmf)
* [Truncated Singular Value Decomposition (TruncatedSVD.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Dimensionality%20Reduction#truncated-singular-value-decomposition-truncated-svd)

**Ensemble Algorithms**
* [Adaboost(AdaBoost.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Ensemble#adaboost-ensemble-learning)
* [Random forest(Random Forest.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Ensemble#random-forest-classification-and-regression)

**Naive Bayes**
* [Naive Bayes(Naive Bayes.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Naive%20Bayes#naive-bayes-classifier)

**Neighbors**
* K-nearest neighbors

**Neural Networks**
* [Kohonen Maps (kohonen maps.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Neural%20Networks#kohonen-maps-self-organizing-maps)
* [Pattern recognition Neural Networks (Pattern Nets.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Neural%20Networks#pattern-recognition-neural-network)
* [Regression Neural Networks (Regressor Nets.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/Neural%20Networks#regression-neural-network)

**Support vector Machine(SVM)**
* [Linear vector Machine (svm.mqh)](https://github.com/MegaJoctan/MALE5/tree/master/SVM#linear-support-vector-machine-svm)


## Installing 

Go to the Include directory and open CMD then run
``` cmd  
git clone https://github.com/MegaJoctan/MALE5.git
```
Or download the zip file from the releases section extract the library, Under MQL5 folder in your MetaEditor, from there paste the MALE5 directory you extracted under the Include folder

## Opening an issue
You can also post bug reports and feature requests (only) in [GitHub issues](https://github.com/MegaJoctan/MALE5/issues).

## Support the Project
If you find this project helpful, Support us by taking one or more of the actions

[BuyMeCoffee](https://www.buymeacoffee.com/omegajoctan)

[OurProducts](https://www.mql5.com/en/users/omegajoctan/seller)

Register to our recommended broker:

[ICMarkets](https://icmarkets.com/?camp=74639)

## Let's work together
Create a personal Job for me on MQL5 | [HIRE ME](https://www.mql5.com/en/job/new?prefered=omegajoctan)


