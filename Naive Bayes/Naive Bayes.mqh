//+------------------------------------------------------------------+
//|                                                  Naive Bayes.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

#include <MALE5\matrix_utils.mqh> 
//+------------------------------------------------------------------+
//|              N  A  I  V  E     B  A  Y  E                        |
//|                                                                  |
//|   suitable for classification of discrete values, that have      |
//|   been load to a matrix using the method ReadCSVEncode from      |
//|   matrix_utils.mqh                                               |
//|                                                                  |
//+------------------------------------------------------------------+

class CNaiveBayes
  {
protected:
   CMatrixutils      matrix_utils; 
   
                     ulong  n;
                     matrix x_matrix;
                     vector y_vector;
                       
                     vector c_prior_proba; //class prior probability
                     vector c_evidence;    //class evidence
                     
                     vector calcProba(vector &v_features);
                     
public:
                     vector classes;       //classes available 
                     
                     CNaiveBayes(void);
                    ~CNaiveBayes(void);
                    
                     void fit(matrix &x, vector &y);
                     int predict(vector &x);
                     vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNaiveBayes::CNaiveBayes(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CNaiveBayes::fit(matrix &x, vector &y)
 {
   x_matrix.Copy(x);
   y_vector.Copy(y); 
   
   classes = matrix_utils.Unique(y_vector);
   
   c_evidence.Resize((ulong)classes.Size());
   
   n = y_vector.Size();
   
   if (n==0) { Print("--> n == 0 | Naive Bayes class failed"); return; }
   
//---

   vector v = {};
   for (ulong i=0; i<c_evidence.Size(); i++)
       {
         v = matrix_utils.Search(y_vector,classes[i]);
         
         c_evidence[i] = (int)v.Size();
       }

//---    
   
   c_prior_proba.Resize(classes.Size());
   
   for (ulong i=0; i<classes.Size(); i++)
      c_prior_proba[i] = c_evidence[i]/(double)n;
   
  
   Print("---> GROUPS ",classes);
   Print("Prior Class Proba ",c_prior_proba,"\nEvidence ",c_evidence);
    
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNaiveBayes::~CNaiveBayes(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CNaiveBayes::predict(vector &x)
 {   
   vector v = calcProba(x);
   
   double sum = v.Sum();
   
   for (ulong i=0; i<v.Size(); i++) //converting the values into probabilities
      v[i] = NormalizeDouble(v[i]/sum,2);       
   
   vector p = v;
   
   #ifdef   DEBUG_MODE
      Print("Probabilities ",p);
   #endif 
   
   return((int)classes[p.ArgMax()]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CNaiveBayes::predict(matrix &x)
 {
  ulong rows = x.Rows();
 
  vector v(rows), pred(rows); 
  
   for (ulong i=0; i<rows; i++)
    { 
       v = x.Row(i);
       pred[i] = predict(v);
    }
    
   return pred;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CNaiveBayes::calcProba(vector &v_features)
 {
    vector proba_v(classes.Size()); //vector to return
    
    if (v_features.Size() != x_matrix.Cols())
      {
         printf("FATAL | Can't calculate probability,  fetures columns size = %d is not equal to x_matrix columns =%d",v_features.Size(),x_matrix.Cols());
         return proba_v;
      }

//---
    
    vector v = {}; 
    
    for (ulong c=0; c<classes.Size(); c++)
      {
        double proba = 1;
          for (ulong i=0; i<x_matrix.Cols(); i++)
            {
                v = x_matrix.Col(i);
                
                int count =0;
                for (ulong j=0; j<v.Size(); j++)
                  {
                     if (v_features[i] == v[j] && classes[c] == y_vector[j])
                        count++;
                  }
                  
                proba *= count==0 ? 1 : count/(double)c_evidence[c]; //do not calculate if there isn't enough evidence'
            }
          
        proba_v[c] = proba*c_prior_proba[c];
     }
     
    return proba_v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|            NORMAL DISTRIBUTION CLASS                             |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

class CNormDistribution
  {

public:
   
   double m_mean; //Assign the value of the mean
   double m_std;  //Assign the value of Variance
   
                     CNormDistribution(void);
                    ~CNormDistribution(void);
                    
                     double PDF(double x); //Probability density function
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

CNormDistribution::CNormDistribution(void)
 {
   
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

CNormDistribution::~CNormDistribution(void)
 {
   ZeroMemory(m_mean);
   ZeroMemory(m_std);
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

double CNormDistribution::PDF(double x)
 {
   double nurm = MathPow((x - m_mean),2)/(2*MathPow(m_std,2));
   nurm = exp(-nurm);
   
   double denorm = 1.0/(MathSqrt(2*M_PI*MathPow(m_std,2)));
      
  return(nurm*denorm);
 }

//+------------------------------------------------------------------+
//|                                                                  |
//|          GAUSSIAN NAIVE BAYES CLASS                              |
//|                                                                  |
//|   Suitable for classification based on features with             |
//|   continuous variables,                                          |
//|                                                                  |
//+------------------------------------------------------------------+

#include <MALE5\preprocessing.mqh>


class CGaussianNaiveBayes
  {
   protected:
   
      CNormDistribution norm_distribution;
      CPreprocessing *normalize_x;

      vector            c_prior_proba; //prior probability
      vector            c_evidence;
      ulong             n;
   
      CMatrixutils       matrix_utils;
      norm_technique     m_norm;
      
      matrix             x_matrix;
      vector             y_vector;
      
      ulong              m_cols;  //columns in x_matrix
      
      bool               during_training;
      vector             calcProba(vector &v_features);
   
   public:              
   
      vector            classes; //Target classes     
             
                        CGaussianNaiveBayes(norm_technique NORM_METHOD=NORM_STANDARDIZATION);
                       ~CGaussianNaiveBayes(void);
                        
                        void fit(matrix &x, vector &y);
                        
                        int predict_bin(vector &x);
                        vector predict_bin(matrix &x);
                        vector predict_proba(vector &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGaussianNaiveBayes::CGaussianNaiveBayes(norm_technique NORM_METHOD=NORM_STANDARDIZATION)
 :m_norm(NORM_METHOD) 
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CGaussianNaiveBayes::fit(matrix &x, vector &y)
 { 
   x_matrix = x;
   y_vector = y;
   
   normalize_x = new CPreprocessing(x_matrix, m_norm);
   
   classes = matrix_utils.Unique(y_vector);
   m_cols = x_matrix.Cols();
    
//---
   
   during_training = true;
   
   c_evidence.Resize((ulong)classes.Size());
   
   n = y_vector.Size();
   
   if (n==0) { Print("---> n == 0 | Gaussian Naive Bayes class failed"); return; }
   
//---
   
   vector v = {};
   for (ulong i=0; i<c_evidence.Size(); i++)
       {          
         v = matrix_utils.Search(y_vector, classes[i]);
         
         c_evidence[i] = (int)v.Size();
       }
   
   c_prior_proba.Resize(classes.Size());
   
   for (ulong i=0; i<classes.Size(); i++)
      c_prior_proba[i] = c_evidence[i]/(double)n;

//---       
   
   Print("---> GROUPS ",classes);
   Print("\n---> Prior_proba ",c_prior_proba," Evidence ",c_evidence);

//---

   during_training = false; 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGaussianNaiveBayes::~CGaussianNaiveBayes(void)
 {
   ZeroMemory(x_matrix);
   ZeroMemory(y_vector);
   delete (normalize_x);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

int CGaussianNaiveBayes::predict_bin(vector &x)
 { 
  vector temp_x = x;
  
  if (!during_training)  
     normalize_x.Normalization(temp_x);
       
   if (temp_x.Size() != m_cols)
     {
       Print("CRITICAL | The given x have different size than the trained x");
       return (-1);
     }
   
   vector p = calcProba(temp_x);
   
   return((int)classes[p.ArgMax()]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CGaussianNaiveBayes::predict_proba(vector &x)
 {
  vector temp_x = x;
  vector ret_v = {};
  
  if (!during_training)  
     normalize_x.Normalization(temp_x);
       
   if (temp_x.Size() != m_cols)
     {
       Print("CRITICAL | The given x have different size than the trained x");
       return (ret_v);
     }
        
   ret_v = calcProba(temp_x);
   
   return (ret_v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CGaussianNaiveBayes::predict_bin(matrix &x)
 {
  ulong rows = x.Rows();
  vector v(rows), pred(rows); 
  
   for (ulong i=0; i<rows; i++)
    { 
       v = x.Row(i);
       
       pred[i] = predict_bin(v);
    }
   
   return pred;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

vector CGaussianNaiveBayes::calcProba(vector &v_features)
 {    
    vector proba_v(classes.Size()); //vector to return
    proba_v.Fill(-1);
    
    if (v_features.Size() != m_cols)
      {
         printf("FATAL | Can't calculate probability, fetures columns size = %d is not equal to x_matrix columns =%d",v_features.Size(),m_cols);
         return proba_v;
      }

//---
    
    vector v = {}; 
    
    for (ulong c=0; c<classes.Size(); c++)
      {
        double proba = 1;
          for (ulong i=0; i<m_cols; i++)
            {
                v = x_matrix.Col(i);
                
                int count =0;
                vector calc_v = {};
                
                for (ulong j=0; j<v.Size(); j++)
                  {
                     if (classes[c] == y_vector[j])
                       {
                         count++;
                         calc_v.Resize(count);
                         
                         calc_v[count-1] = v[j];
                       }
                  } 
                
                norm_distribution.m_mean = calc_v.Mean(); //Assign these to Gaussian Normal distribution
                norm_distribution.m_std = calc_v.Std();   
                 
                
                #ifdef DEBUG_MODE
                  printf("mean %.5f std %.5f ",norm_distribution.m_mean,norm_distribution.m_std);
                #endif 
                
                proba *= count==0 ? 1 : norm_distribution.PDF(v_features[i]); //do not calculate if there isn't enought evidence'
            }
          
        proba_v[c] = proba*c_prior_proba[c]; //Turning the probability density into probability
        
        #ifdef DEBUG_MODE
         Print(">> Proba ",proba," prior proba ",c_prior_proba);
        #endif 
     }

//--- Normalize probabilities
    
    proba_v = proba_v / proba_v.Sum();
    
    return proba_v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+