//+------------------------------------------------------------------+
//|                                                      metrics.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

#include <MALE5\matrix_utils.mqh>

#define ZeroDivide(value) value==0?1.0:(double)value
              
struct confusion_matrix_struct
{
    double accuracy;
    vector<double> precision;
    vector<double> recall;
    vector<double> f1_score;
    vector<double> specificity;
    vector<double> support;
   
    vector<double> avg;
    vector<double> w_avg;
   
};

class CMetrics
  {
CMatrixutils matrix_utils;

protected:
   int SearchPatterns(vector &True, int value_A, vector &B, int value_B);
   
//-- From matrix utility class
   
public:
                     CMetrics(void);
                    ~CMetrics(void);
                    
                  //--- Regression metrics
                  
                    double r_squared(vector &True, vector &Pred); 
                    double adjusted_r(vector &True, vector &Pred,uint indep_vars=1);
                    
                    double rss(vector &True, vector &Pred);
                    double mse(vector &True, vector &Pred);
                    double rmse(vector &True, vector &Pred);
                    double mae(vector &True, vector &Pred);
                 
                 //--- Classification metrics
                    
                    double accuracy_score(vector &True, vector &Pred);
                    confusion_matrix_struct confusion_matrix(vector &True, vector &Pred, bool report_show=true);
                    
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMetrics::CMetrics(void)
  {
  
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CMetrics::~CMetrics(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::r_squared(vector &True, vector &P)
 { 
   return(P.RegressionMetric(True, REGRESSION_R2));      
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

double CMetrics::adjusted_r(vector &True,vector &Pred,uint indep_vars=1)
 {
   if (True.Size() != Pred.Size())
      {
         Print(__FUNCTION__," Vector True and P are not equal in size ");
         return(0);
      }
      
   double r2 = r_squared(True,Pred);
   ulong N = Pred.Size();
   
   return(1-( (1-r2)*(N-1) )/(N - indep_vars -1));
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
confusion_matrix_struct  CMetrics::confusion_matrix(vector &True,vector &Pred,bool report_show=true)
 {     
    ulong TP=0, TN=0, FP=0, FN=0;
    
    vector classes = matrix_utils.Unique(True);
    
    matrix conf_m(classes.Size(),classes.Size());
    conf_m.Fill(0); 
    
    vector row(classes.Size());       
    
    vector conf_v(ulong(MathPow(classes.Size(),2)));
    
    if (True.Size() != Pred.Size())
      {
         Print("True and Pred vectors are not same in size ");
         //return confusion_matrix_struct;
      }

//---  
   
    for (ulong i=0; i<classes.Size(); i++)
      {  
       ulong col_=0, row_=0; 
         
//--- 
          for (ulong j=0; j<classes.Size(); j++)
             {                
                 conf_m[i][j] = SearchPatterns(True,(int)classes[i],Pred,(int)classes[j]);                  
             }
      }
   
   confusion_matrix_struct confusion_mat;
      
//--- METRICS
   
   vector diag = conf_m.Diag();
   confusion_mat.accuracy = NormalizeDouble(diag.Sum()/conf_m.Sum(),3);

//--- precision
   
   confusion_mat.precision = Pred.ClassificationMetric(True, CLASSIFICATION_PRECISION, AVERAGE_BINARY);
   
//--- recall
   
   confusion_mat.recall = Pred.ClassificationMetric(True, CLASSIFICATION_RECALL, AVERAGE_BINARY);
   
//--- specificity

   confusion_mat.specificity = Pred.ClassificationMetric(True, CLASSIFICATION_BALANCED_ACCURACY, AVERAGE_BINARY);

//--- f1 score
   
   confusion_mat.f1_score = Pred.ClassificationMetric(True, CLASSIFICATION_F1, AVERAGE_BINARY);

//--- support
   
   confusion_mat.support.Resize(classes.Size());
   
   vector row_v;
   for (ulong i=0; i<classes.Size(); i++)
     {
         row_v = conf_m.Row(i);
         confusion_mat.support[i] = NormalizeDouble(MathIsValidNumber(row_v.Sum())?row_v.Sum():0,8);
     }
     
   int total_size = (int)conf_m.Sum();
   
//--- Avg and w avg
   
   confusion_mat.avg.Resize(5);
   confusion_mat.w_avg.Resize(5);
     
    confusion_mat.avg[0] = confusion_mat.precision.Mean();       
    
    confusion_mat.avg[1] = confusion_mat.recall.Mean();
    confusion_mat.avg[2] = confusion_mat.specificity.Mean();
    confusion_mat.avg[3] = confusion_mat.f1_score.Mean();
    
    confusion_mat.avg[4] = total_size;
   
//--- w avg
    
   vector support_prop = confusion_mat.support/(double)total_size;
   
   vector c = confusion_mat.precision * support_prop;
   confusion_mat.w_avg[0] = c.Sum();
   
   c = confusion_mat.recall * support_prop;
   confusion_mat.w_avg[1] = c.Sum();
   
   c = confusion_mat.specificity * support_prop;
   confusion_mat.w_avg[2] = c.Sum();
   
   c = confusion_mat.f1_score * support_prop;
   confusion_mat.w_avg[3] = c.Sum();
   
   confusion_mat.w_avg[4] = (int)total_size;
   
//--- Report
   
   if (report_show)
    {
      string report = "\n_\t\t\t\tPrecision \tRecall \tSpecificity \tF1 score \tSupport";
      
      for (ulong i=0; i<classes.Size(); i++)
         {
           report += "\n\t"+string(classes[i]);
             //for (ulong j=0; j<3; j++)
             
               report += StringFormat("\t\t\t %.2f \t\t\t %.2f \t\t\t %.2f \t\t\t\t\t %.2f \t\t\t %.1f",confusion_mat.precision[i],confusion_mat.recall[i],confusion_mat.specificity[i],confusion_mat.f1_score[i],confusion_mat.support[i]);
         }
     
      report += StringFormat("\n\nAccuracy\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%.2f\n",confusion_mat.accuracy);
       
      report += StringFormat("Average \t %.2f \t\t %.2f \t\t %.2f \t\t\t\t %.2f \t\t %.1f",confusion_mat.avg[0],confusion_mat.avg[1],confusion_mat.avg[2],confusion_mat.avg[3],confusion_mat.avg[4]);
      report += StringFormat("\nW Avg \t\t\t %.2f \t\t %.2f \t\t %.2f \t\t\t\t %.2f \t\t %.1f",confusion_mat.w_avg[0],confusion_mat.w_avg[1],confusion_mat.w_avg[2],confusion_mat.w_avg[3],confusion_mat.w_avg[4]);
          
      Print("Confusion Matrix\n",conf_m);    
      Print("\nClassification Report\n",report);
   }
//---
      
   return (confusion_mat);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::rss(vector &True,vector &Pred)
 {
   vector c = True-Pred;
   c = MathPow(c,2);
   
   return (c.Sum()); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::mse(vector &True,vector &Pred)
 {
   vector c = True - Pred;
   c = MathPow(c,2);
   
   return(c.Sum()/c.Size()); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::accuracy_score(vector &True,vector &Pred)
 {
   return this.confusion_matrix(True,Pred,false).accuracy;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CMetrics::SearchPatterns(vector &True, int value_A, vector &B, int value_B)
 {
   int count=0;
  
   for (ulong i=0; i<True.Size(); i++)
     {
       if (True[i] == value_A && B[i] == value_B)
         { 
           count++;
         }
     }
      
    return count;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::rmse(vector &True,vector &Pred)
 {
   return Pred.RegressionMetric(True, REGRESSION_RMSE);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::mae(vector &True, vector &Pred)
 {
   return Pred.RegressionMetric(True, REGRESSION_MAE);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

