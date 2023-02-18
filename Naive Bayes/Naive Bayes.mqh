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
                     matrix XMatrix;
                     vector YVector;
                       
                     vector c_prior_proba; //class prior probability
                     vector c_evidence;    //class evidence
                     
                     vector calcProba(vector &v_features);
                     
public:
                     vector classes;       //classes available 
                     
                     CNaiveBayes(matrix &x_matrix, vector &y_vector);
                    ~CNaiveBayes(void);
                    
                     int NaiveBayes(vector &x_vector);
                     vector NaiveBayes(matrix &x_matrix);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNaiveBayes::CNaiveBayes(matrix &x_matrix, vector &y_vector)
 {
 
   XMatrix.Copy(x_matrix);
   YVector.Copy(y_vector); 
   
   classes = matrix_utils.Classes(YVector);
   
   c_evidence.Resize((ulong)classes.Size());
   
   n = YVector.Size();
   
   if (n==0) { Print("--> n == 0 | Naive Bayes class failed"); return; }
   
//---

   vector v = {};
   for (ulong i=0; i<c_evidence.Size(); i++)
       {
         v = matrix_utils.Search(YVector,(int)classes[i]);
         
         c_evidence[i] = (int)v.Size();
       }

//---
    
   
   c_prior_proba.Resize(classes.Size());
   
   for (ulong i=0; i<classes.Size(); i++)
      c_prior_proba[i] = c_evidence[i]/(double)n;
   
    
  
   //#ifdef DEBUG_MODE
      Print("---> GROUPS ",classes);
      Print("Prior Class Proba ",c_prior_proba,"\nEvidence ",c_evidence);
   //#endif 
    
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CNaiveBayes::~CNaiveBayes(void)
 {
   ZeroMemory(XMatrix);
   ZeroMemory(YVector); 
   ZeroMemory(c_prior_proba);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CNaiveBayes::NaiveBayes(vector &x_vector)
 {   
   vector v = calcProba(x_vector);
   
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
vector CNaiveBayes::NaiveBayes(matrix &x_matrix)
 {
  ulong rows = x_matrix.Rows();
 
  vector v(rows), pred(rows); 
  
   for (ulong i=0; i<rows; i++)
    { 
       v = x_matrix.Row(i);
       pred[i] = NaiveBayes(v);
    }
    
   return pred;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CNaiveBayes::calcProba(vector &v_features)
 {
    vector proba_v(classes.Size()); //vector to return
    
    if (v_features.Size() != XMatrix.Cols())
      {
         printf("FATAL | Can't calculate probability,  fetures columns size = %d is not equal to XMatrix columns =%d",v_features.Size(),XMatrix.Cols());
         return proba_v;
      }

//---
    
    vector v = {}; 
    
    for (ulong c=0; c<classes.Size(); c++)
      {
        double proba = 1;
          for (ulong i=0; i<XMatrix.Cols(); i++)
            {
                v = XMatrix.Col(i);
                
                int count =0;
                for (ulong j=0; j<v.Size(); j++)
                  {
                     if (v_features[i] == v[j] && classes[c] == YVector[j])
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
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//|                                                                  |
//|            NORMAL DISTRIBUTION CLASS                             |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
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

class CGaussianNaiveBayes
  {
   protected:
   
      CNormDistribution norm_distribution;
      
      vector            c_prior_proba; //prior probability
      vector            c_evidence;
      ulong             n;
      
   protected:
   
      CMatrixutils       matrix_utils;
      
      matrix             XMatrix;
      vector             YVector; 
      ulong              m_cols;  //columns in XMatrix
      
      
      vector             calcProba(vector &v_features);
   
   public:              
   
      vector            classes; //Target classes            
                        CGaussianNaiveBayes(matrix &x_matrix, vector &y_vector);
                       ~CGaussianNaiveBayes(void);
                        
                        int GaussianNaiveBayes(vector &x_features);
                        vector GaussianNaiveBayes(matrix &x_matrix);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGaussianNaiveBayes::CGaussianNaiveBayes(matrix &x_matrix, vector &y_vector)
 { 
      
   XMatrix.Copy(x_matrix);
   YVector.Copy(y_vector);
   
   classes = matrix_utils.Classes(YVector);
   
   m_cols = XMatrix.Cols();
    
//---
   
   c_evidence.Resize((ulong)classes.Size());
   
   n = YVector.Size();
   
   if (n==0) { Print("---> n == 0 | Gaussian Naive Bayes class failed"); return; }
   
//---

   vector v = {};
   for (ulong i=0; i<c_evidence.Size(); i++)
       {
         v = matrix_utils.Search(YVector,(int)classes[i]);
         
         c_evidence[i] = (int)v.Size();
       }

   
   c_prior_proba.Resize(classes.Size());
   
   for (ulong i=0; i<classes.Size(); i++)
      c_prior_proba[i] = c_evidence[i]/(double)n;

//---       
   
   
   //#ifdef DEBUG_MODE
      Print("---> GROUPS ",classes);
      Print("\n---> Prior_proba ",c_prior_proba," Evidence ",c_evidence);
   //#endif 
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGaussianNaiveBayes::~CGaussianNaiveBayes(void)
 {
   ZeroMemory(XMatrix);
   ZeroMemory(YVector);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

int CGaussianNaiveBayes::GaussianNaiveBayes(vector &x_features)
 { 
   if (x_features.Size() != m_cols)
     {
       Print("CRITICAL | The given x_features have different size than the trained x_features");
       return (-1);
     }
     
   vector p = calcProba(x_features);
  
   
   return((int)classes[p.ArgMax()]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CGaussianNaiveBayes::GaussianNaiveBayes(matrix &x_matrix)
 {
  ulong rows = x_matrix.Rows();
 
  vector v(rows), pred(rows); 
  
   for (ulong i=0; i<rows; i++)
    { 
       v = x_matrix.Row(i);
       pred[i] = GaussianNaiveBayes(v);
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
    
    if (v_features.Size() != XMatrix.Cols())
      {
         printf("FATAL | Can't calculate probability, fetures columns size = %d is not equal to XMatrix columns =%d",v_features.Size(),XMatrix.Cols());
         return proba_v;
      }

//---
    
    vector v = {}; 
    
    for (ulong c=0; c<classes.Size(); c++)
      {
        double proba = 1;
          for (ulong i=0; i<XMatrix.Cols(); i++)
            {
                v = XMatrix.Col(i);
                
                int count =0;
                vector calc_v = {};
                
                for (ulong j=0; j<v.Size(); j++)
                  {
                     if (classes[c] == YVector[j])
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
     
    return proba_v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+