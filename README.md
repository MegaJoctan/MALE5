# MALE5: Machine Learning for MQL5

![MALE5](MALE5.png)

This repository hosts the development of the MALE 5 library

## About MALE5

MALE5 is a machine learning repository for creating trading systems in the c++ like, MQL5 programming language.
It was developed to help build machine learning based trading robots, effortlessly in the [MetaTrader5](https://www.metatrader5.com/en/automated-trading/metaeditor) platform

This Library is:

-   **Simple to use** You can literly start building your system once you call class constructor
-   **Flexible** You can use it any program scripts, Indicator, EA's
-   **Resources cheap** It doesn't consume a lot of memory neither the CPU, takes short time intervals to train

**ML Algorithms Available:**

*currently there are not many algorithms ready as I am a solodev finding a way through*

-   Linear Regression
-   Logistic Regression
-   Polynomial Regression
-   ridge  & Laso Regression
-   Classification decision tree
-   Naive bayes
-   FeedForward Neural Network
-   KNN nearest neighbors

**Clustering techniques | Unsupervised Learning:**

-   KNN clustering 

## Installing 

Create a directory with the name MALE5 under your include directory in Metaeditor then open the command terminal in that directory and type 

``` cmd
    git clone https://github.com/MegaJoctan/MALE5.git
```

## Using the Library

Once the Library is installed and everything is set in the right directory, here is how to install and use the models; Look at the Linear regression example

``` MQL5
#include <MALE5\matrix_utils.mqh>
#include <MALE5\Linear Regression\Linear Regression.mqh>

CLinearRegression *Linear_reg;
CMatrixutils matrix_utils;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
//---

    matrix Matrix = matrix_utils.ReadCsv("NASDAQ_DATA.csv"); 

    Linear_reg = new CLinearRegression(Matrix);
    
    double acc =0; //Model accuracy
    
    Linear_reg.LRModelPred(Matrix,acc);
    
    Print("Trained Model Accuracy ",acc);
}
```

## Opening an issue

You can also post **bug reports and feature requests** (only)
in [GitHub issues](https://github.com/MegaJoctan/MALE5/issues).

## Contributing 

I welcome and appreciate contributions: feel free to contact me anytime at omegajoctan@gmail.com

## Donate

[Buy Me Coffee](https://www.buymeacoffee.com/omegajoctan)