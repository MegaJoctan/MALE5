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
   ulong   rows, cols;
   matrix            component_matrix;
   vector            eigen_vectors;
   
   matrix            EighVars(matrix &_matrix);
   
public:
                     Cpca(matrix &Matrix);
                    ~Cpca(void);
                    
                     matrix pca_scores;
                     vector pca_scores_coefficients;
                     matrix pca_scores_standardized;
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
Cpca::Cpca(matrix &Matrix)
 { 
   rows = Matrix.Rows(); 
   cols = Matrix.Cols();
   
   pre_processing = new CPreprocessing(Matrix, NORM_STANDARDIZATION);
   
   Print("Standardized data\n",Matrix);
   
   matrix Cova = Matrix.Cov(false);
   
   Print("Covariances\n", Cova);
   
   if (!Cova.Eig(component_matrix, eigen_vectors))
      Print("Failed to get the Component matrix matrix & Eigen vectors");
   
   Print("\nComponent matrix\n",component_matrix,"\nEigen Vectors\n",eigen_vectors);
   
   pca_scores = Matrix.MatMul(component_matrix);

   Print("PCA SCORES\n",pca_scores);
   
//---

   pca_scores_coefficients.Resize(cols);
   vector v_row;
   
   for (ulong i=0; i<cols; i++)
     {
       v_row = pca_scores.Col(i);
       
       pca_scores_coefficients[i] = v_row.Var(); //variance of the pca scores
     }
   
   Print("SCORES COEFF ",pca_scores_coefficients); 
   
//---

   pca_scores_standardized.Copy(pca_scores);
   
   delete (pre_processing);
   
   pre_processing = new CPreprocessing(pca_scores_standardized, NORM_STANDARDIZATION);
   
   Print("PCA SCORES | STANDARDIZED\n",pca_scores_standardized);
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