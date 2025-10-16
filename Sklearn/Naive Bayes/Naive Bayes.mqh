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

#include <MALE5\MatrixExtend.mqh> 
//+------------------------------------------------------------------+
//|              N  A  I  V  E     B  A  Y  E                        |
//|                                                                  |
//|   suitable for classification of discrete values, that have      |
//|   been load to a matrix using the method ReadCSVEncode from      |
//|   MatrixExtend::mqh                                               |
//|                                                                  |
//+------------------------------------------------------------------+

class CNaiveBayes
  {
protected:
                     uint n_features;
                     vector y_target;
                     
                     vector class_proba; //prior class probability
                     vector features_proba; //features probability
                       
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
  ulong samples = x.Rows(),
        features = x.Cols();
  
  vector unique = MatrixExtend::Unique_count(y);
  
  this.class_proba = unique / samples;
  
  if (MQLInfoInteger(MQL_DEBUG))
    Print("class probabilities: ",class_proba);
  
    
 
/*
   y_target = y;
   n_features = x.Cols();
   
   classes = MatrixExtend::Unique(y);
   
   c_evidence.Resize((ulong)classes.Size());
   
   n = y.Size();
   
   if (n==0) { Print("--> n == 0 | Naive Bayes class failed"); return; }
   
//---

   vector v = {};
   for (ulong i=0; i<c_evidence.Size(); i++)
       {
         v = MatrixExtend::Search(y,classes[i]);
         
         c_evidence[i] = (int)v.Size();
       }

//---    
   
   c_prior_proba.Resize(classes.Size());
   
   for (ulong i=0; i<classes.Size(); i++)
      c_prior_proba[i] = c_evidence[i]/(double)n;
   
  
   Print("---> GROUPS ",classes);
   Print("Prior Class Proba ",c_prior_proba,"\nEvidence ",c_evidence);
*/  
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
/*
vector CNaiveBayes::calcProba(vector &v_features)
 {
    vector proba_v(classes.Size()); //vector to return
    
    if (v_features.Size() != n_features)
      {
         printf("FATAL | Can't calculate probability,  fetures columns size = %d is not equal to x_matrix columns =%d",v_features.Size(),n_features);
         return proba_v;
      }

//---
    
    vector v = {}; 
    
    for (ulong c=0; c<classes.Size(); c++)
      {
        double proba = 1;
          for (ulong i=0; i<n_features; i++)
            {
                v = x_matrix.Col(i);
                
                int count =0;
                for (ulong j=0; j<v.Size(); j++)
                  {
                     if (v_features[i] == v[j] && classes[c] == y[j])
                        count++;
                  }
                  
                proba *= count==0 ? 1 : count/(double)c_evidence[c]; //do not calculate if there isn't enough evidence'
            }
          
        proba_v[c] = proba*c_prior_proba[c];
     }
     
    return proba_v;
 }*/
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

/*
class CGaussianNaiveBayes
  {
   protected:
   
      CNormDistribution norm_distribution;

      vector            c_prior_proba; //prior probability
      vector            c_evidence;
      ulong             n;
      
      ulong              m_cols;  //columns in x_matrix
      vector             calcProba(vector &v_features);
   
   public:              
   
      vector            classes; //Target classes     
             
                        CGaussianNaiveBayes(void);
                       ~CGaussianNaiveBayes(void);
                        
                        void fit(matrix &x, vector &y);
                        
                        int predict_bin(vector &x);
                        vector predict_bin(matrix &x);
                        vector predict_proba(vector &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CGaussianNaiveBayes::CGaussianNaiveBayes(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CGaussianNaiveBayes::fit(matrix &x, vector &y)
 { 
   
   classes = MatrixExtend::Unique(y);
   m_cols = n_features;
    
//---
   
   c_evidence.Resize((ulong)classes.Size());
   
   n = y.Size();
   
   if (n==0) { Print("---> n == 0 | Gaussian Naive Bayes class failed"); return; }
   
//---
   
   vector v = {};
   for (ulong i=0; i<c_evidence.Size(); i++)
       {          
         v = MatrixExtend::Search(y, classes[i]);
         
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
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

int CGaussianNaiveBayes::predict_bin(vector &x)
 {     
   if (x.Size() != m_cols)
     {
       Print("CRITICAL | The given x have different size than the trained x");
       return (-1);
     }
   
   vector p = calcProba(x);
   
   return((int)classes[p.ArgMax()]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CGaussianNaiveBayes::predict_proba(vector &x)
 {
  vector x = x;
  vector ret_v = {};
  
   if (x.Size() != m_cols)
     {
       Print("CRITICAL | The given x have different size than the trained x");
       return (ret_v);
     }
        
   ret_v = calcProba(x);
   
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
                     if (classes[c] == y[j])
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
*/