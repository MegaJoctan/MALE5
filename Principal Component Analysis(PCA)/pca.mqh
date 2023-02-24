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
#include <MALE5\MqPlotLib\plots.mqh>

enum criterion
  {
    CRITERION_VARIANCE,
    CRITERION_KAISER,
    CRITERION_SCREE_PLOT
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class Cpca
  {
CPreprocessing       *pre_processing;
CPlots   plt;

protected:
   ulong   rows, cols;
   matrix            component_matrix;
   vector            eigen_vectors;
   
   void              Swap(double &var1, double &var2);
   void              SortAscending(vector &v);
   bool              VectorToArray(const vector &v,double &arr[]);
   
public:
                     Cpca(matrix &Matrix);
                    ~Cpca(void);
                    
                     matrix pca_scores;
                     vector pca_scores_coefficients;
                     matrix pca_scores_standardized;
                     
                     matrix ExtractComponents(criterion CRITERION_);
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
matrix Cpca::ExtractComponents(criterion CRITERION_)
 {

  vector vars = pca_scores_coefficients;   
  vector vars_percents = (vars/(double)vars.Sum())*100.0;
  
//--- for Kaiser

  double vars_mean = pca_scores_coefficients.Mean();

//--- for scree
   
   double x[], y[];
   
//---

  matrix PCAS = {};
  
  double sum=0;
  ulong  max;
  vector v_cols = {};
   
   switch(CRITERION_)
     {
  
      case  CRITERION_VARIANCE: 
      
        Print("vars percentages ",vars_percents);       
        
         for (int i=0, count=0; i<(int)cols; i++)
           { 
             count++;
             
              max = vars_percents.ArgMax();
              sum += vars_percents[max];
              
              vars_percents[max] = 0; 
              
              v_cols.Resize(count);
              v_cols[count-1] = (int)max;
                   
              if (sum >= 90.0) break;
           }
         
         PCAS.Resize(rows, v_cols.Size());
         
         for (ulong i=0; i<v_cols.Size(); i++)
            PCAS.Col(pca_scores.Col((ulong)v_cols[i]), i);
         
        break;
      case  CRITERION_KAISER:
      
       Print("var ",vars," scores mean ",vars_mean);
       
       vars = pca_scores_coefficients;
        for (ulong i=0, count=0; i<cols; i++)
           if (vars[i] > vars_mean)
             {
               count++;
       
               PCAS.Resize(rows, count);
               
               PCAS.Col(pca_scores.Col(i), count-1);
             }           
           
        break;
      case  CRITERION_SCREE_PLOT:
         
         v_cols.Resize(cols);
         
         for (ulong i=0; i<v_cols.Size(); i++)
             v_cols[i] = (int)i+1;
             
         
          vars = pca_scores_coefficients;
          
          VectorToArray(v_cols, x);
          VectorToArray(vars, y);
          
          plt.ScatterCurvePlots("Scree plot",x,y,"variance","PCA","Variance");

//---

       vars = pca_scores_coefficients;
        for (ulong i=0, count=0; i<cols; i++)
           if (vars[i] > vars_mean)
             {
               count++;
       
               PCAS.Resize(rows, count);
               
               PCAS.Col(pca_scores.Col(i), count-1);
             }    
             
        break;
     } 
   return (PCAS);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void Cpca::Swap(double &var1,double &var2)
 {
   double temp_1 = var1, temp2=var2;
   
   var1 = temp2;
   var2 = temp_1;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Cpca::SortAscending(vector &v)
 { 
    ulong n = v.Size();
    for (ulong i = 0; i < n - 1; i++)
      {
        ulong minIndex = i;
        for (ulong j = i + 1; j < n; j++)
          {
            if (v[j] < v[minIndex]) {
                minIndex = j;
           }
      }
      
      Swap(v[i], v[minIndex]);
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool Cpca::VectorToArray(const vector &v,double &arr[])
  {
   ArrayResize(arr,(int)v.Size());

   if(ArraySize(arr) == 0)
      return(false);

   for(ulong i=0; i<v.Size(); i++)
      arr[i] = v[i];

   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
