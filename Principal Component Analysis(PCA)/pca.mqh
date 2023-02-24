//+------------------------------------------------------------------+
//|                                                          pca.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//|         Principle Component Analysis Library                     |
//+------------------------------------------------------------------+
#include <MALE5\preprocessing.mqh>

class Cpca
  {
CPreprocessing       *pre_processing;

protected:
   matrix            component_matrix;
   vector            eigen_vectors;
   
   matrix            EighVars(matrix &_matrix);
public:
                     Cpca(matrix &Matrix);
                    ~Cpca(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
Cpca::Cpca(matrix &Matrix)
 { 
   pre_processing = new CPreprocessing(Matrix, NORM_STANDARDIZATION);
   
   Print("Standardized data\n",Matrix);
   
   matrix Cova = Matrix.Cov(false);
   
   Print("Covariances\n", Cova);
   
   if (!Cova.Eig(component_matrix, eigen_vectors))
      Print("Failed to get the Component matrix matrix & Eigen vectors");
   
   Print("\nComponent matrix\n",component_matrix,"\nEigen Vectors\n",eigen_vectors);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
Cpca::~Cpca(void)
 {
   delete (pre_processing);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
