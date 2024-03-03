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

#include <MALE5\MatrixExtend.mqh>
#include <MALE5\MqPlotLib\plots.mqh>

struct roc_curve_struct
 {
   vector TPR,
          FPR, 
          Thresholds;
 };

struct confusion_matrix_struct
 { 
   matrix MATRIX;
   vector CLASSES;
   vector TP, 
          TN, 
          FP, 
          FN;
 };
 
enum regression_metrics
{
   METRIC_R_SQUARED,   // R-squared
   METRIC_ADJUSTED_R,  // Adjusted R-squared
   METRIC_RSS,         // Residual Sum of Squares
   METRIC_MSE,         // Mean Squared Error
   METRIC_RMSE,        // Root Mean Squared Error
   METRIC_MAE          // Mean Absolute Error
};

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class Metrics
  {
protected:   
   static int SearchPatterns(const vector &True, int value_A, const vector &B, int value_B);

   static confusion_matrix_struct confusion_matrix(const vector &True, const vector &Preds);
   
public:

   Metrics(void);
   ~Metrics(void);

   //--- Regression metrics

   static double r_squared(const vector &True, const vector &Pred);
   static double adjusted_r(const vector &True, const vector &Pred, uint indep_vars = 1);

   static double rss(const vector &True, const vector &Pred);
   static double mse(const vector &True, const vector &Pred);
   static double rmse(const vector &True, const vector &Pred);
   static double mae(const vector &True, const vector &Pred);
   
   static double RegressionMetric(const vector &True, const vector &Pred, regression_metrics METRIC_);

   //--- Classification metrics

   static double accuracy_score(const vector &True, const vector &Pred);
   
   static vector accuracy(const vector &True, const vector &Preds);
   static vector precision(const vector &True, const vector &Preds);
   static vector recall(const vector &True, const vector &Preds);
   static vector f1_score(const vector &True, const vector &Preds);
   static vector specificity(const vector &True, const vector &Preds);
   
   static roc_curve_struct roc_curve(const vector &True, const vector &Preds, bool show_roc_curve=false);
   static void classification_report(const vector &True, const vector &Pred, bool show_roc_curve=false);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
Metrics::Metrics(void)
  {

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
Metrics::~Metrics(void)
  {

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Metrics::r_squared(const vector &True, const vector &Pred)
  {
   return(Pred.RegressionMetric(True, REGRESSION_R2));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Metrics::adjusted_r(const vector &True, const vector &Pred, uint indep_vars = 1)
  {
   if(True.Size() != Pred.Size())
     {
      Print(__FUNCTION__, " Vector True and P are not equal in size ");
      return(0);
     }

   double r2 = r_squared(True, Pred);
   ulong N = Pred.Size();

   return(1 - ((1 - r2) * (N - 1)) / (N - indep_vars - 1));
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
confusion_matrix_struct Metrics::confusion_matrix(const vector &True, const vector &Preds)
 {
  confusion_matrix_struct confusion_matrix; 
   
  vector classes = MatrixExtend::Unique(True);
  confusion_matrix.CLASSES = classes;
  
//--- Fill the confusion matrix
   
   matrix MATRIX(classes.Size(), classes.Size());
   MATRIX.Fill(0.0);
   
   for(ulong i = 0; i < classes.Size(); i++)
      for(ulong j = 0; j < classes.Size(); j++)
         MATRIX[i][j] = SearchPatterns(True, (int)classes[i], Preds, (int)classes[j]);
   
   confusion_matrix.MATRIX = MATRIX;
   confusion_matrix.TP = MATRIX.Diag();
   confusion_matrix.FP = MATRIX.Sum(0) - confusion_matrix.TP;
   confusion_matrix.FN = MATRIX.Sum(1) - confusion_matrix.TP;
   confusion_matrix.TN = MATRIX.Sum() - (confusion_matrix.TP + confusion_matrix.FP + confusion_matrix.FN);
     
   return confusion_matrix;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector Metrics::accuracy(const vector &True,const vector &Preds)
 {
  confusion_matrix_struct conf_m = confusion_matrix(True, Preds);
  
  return (conf_m.TP + conf_m.TN) / conf_m.MATRIX.Sum();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector Metrics::precision(const vector &True,const vector &Preds)
 {
   confusion_matrix_struct conf_m = confusion_matrix(True, Preds);

   return conf_m.TP / (conf_m.TP + conf_m.FP + DBL_EPSILON); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector Metrics::f1_score(const vector &True,const vector &Preds)
 {
   vector precision = precision(True, Preds);
   vector recall = recall(True, Preds);
   
   return 2 * precision * recall / (precision + recall + DBL_EPSILON); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector Metrics::recall(const vector &True,const vector &Preds)
 {
   confusion_matrix_struct conf_m = confusion_matrix(True, Preds);

   return conf_m.TP / (conf_m.TP + conf_m.FN + DBL_EPSILON); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector Metrics::specificity(const vector &True,const vector &Preds)
 {
   confusion_matrix_struct conf_m = confusion_matrix(True, Preds);

   return conf_m.TN / (conf_m.TN + conf_m.FP + DBL_EPSILON); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
roc_curve_struct Metrics::roc_curve(const vector &True,const vector &Preds, bool show_roc_curve=false)
 {
   roc_curve_struct roc;
   confusion_matrix_struct conf_m = confusion_matrix(True, Preds);
   
   roc.TPR = recall(True, Preds);
   roc.FPR = conf_m.FP / (conf_m.FP + conf_m.TN + DBL_EPSILON);
   
   if (show_roc_curve)
   {
      CPlots plt;
      plt.Plot("Roc Curve",roc.FPR,roc.TPR,"roc_curve","False Positive Rate(FPR)","True Positive Rate(TPR)");
      
      while (MessageBox("Close or Cancel ROC CURVE to proceed","Roc Curve",MB_OK)<0)
       Sleep(1);
   } 
   
   return roc;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Metrics::accuracy_score(const vector &True, const vector &Preds)
  {
   confusion_matrix_struct conf_m = confusion_matrix(True, Preds);
   
   return conf_m.MATRIX.Diag().Sum() / (conf_m.MATRIX.Sum() + DBL_EPSILON);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void Metrics::classification_report(const vector &True, const vector &Pred, bool show_roc_curve=false)
  {
  
  vector accuracy = accuracy(True, Pred);
  vector precision = precision(True, Pred);
  vector specificity = specificity(True, Pred);
  vector recall = recall(True, Pred);
  vector f1_score = f1_score(True, Pred); 
  
  double acc = accuracy_score(True, Pred);
  
  confusion_matrix_struct conf_m = confusion_matrix(True, Pred);
  
//--- support
   
   ulong size = conf_m.MATRIX.Rows();
   
   vector support(size);
   
   for(ulong i = 0; i < size; i++)
      support[i] = NormalizeDouble(MathIsValidNumber(conf_m.MATRIX.Row(i).Sum()) ? conf_m.MATRIX.Row(i).Sum() : 0, 8);

   int total_size = (int)conf_m.MATRIX.Sum();

//--- Avg and w avg
   
   vector avg, w_avg;
   avg.Resize(5);
   w_avg.Resize(5);

   avg[0] = precision.Mean();

   avg[1] = recall.Mean();
   avg[2] = specificity.Mean();
   avg[3] = f1_score.Mean();

   avg[4] = total_size;

//--- w avg

   vector support_prop = support / double(total_size + 1e-10);

   vector c = precision * support_prop;
   w_avg[0] = c.Sum();

   c = recall * support_prop;
   w_avg[1] = c.Sum();

   c = specificity * support_prop;
   w_avg[2] = c.Sum();

   c = f1_score * support_prop;
   w_avg[3] = c.Sum();

   w_avg[4] = (int)total_size;

//--- Report

      string report = "\n[CLS][ACC] \t\t\t\t\tprecision \trecall \tspecificity \tf1 score \tsupport";

      for(ulong i = 0; i < size; i++)
        {
         report += "\n\t[" + string(conf_m.CLASSES[i])+"]["+DoubleToString(accuracy[i], 2)+"]";
         //for (ulong j=0; j<3; j++)

         report += StringFormat("\t\t\t\t\t %.2f \t\t\t\t\t %.2f \t\t\t\t\t %.2f \t\t\t\t\t %.2f \t\t\t\t %d", precision[i], recall[i], specificity[i], f1_score[i], (int)support[i]);
        }
      
      report += "\n";
      report += StringFormat("\naccuracy\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t %.2f \t\t\t\t %d",acc,(int)conf_m.MATRIX.Sum());
      
      report += StringFormat("\naverage\t\t\t\t\t\t\t\t\t %.2f \t\t\t\t\t %.2f \t\t\t\t\t %.2f \t\t\t\t\t %.2f \t\t\t\t %d", avg[0], avg[1], avg[2], avg[3], (int)avg[4]);
      report += StringFormat("\nWeighed avg\t\t\t \t %.2f \t\t\t\t\t %.2f \t\t\t\t\t %.2f \t\t\t\t\t %.2f \t\t\t\t %d", w_avg[0], w_avg[1], w_avg[2], w_avg[3], (int)w_avg[4]);

      Print("Confusion Matrix\n", conf_m.MATRIX);
      Print("\nClassification Report\n", report);
      
      roc_curve(True, Pred, show_roc_curve);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Metrics::rss(const vector &True, const vector &Pred)
  {
   vector c = True - Pred;
   c = MathPow(c, 2);

   return (c.Sum());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Metrics::mse(const vector &True, const vector &Pred)
  {
   vector c = True - Pred;
   c = MathPow(c, 2);

   return(c.Sum() / c.Size());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int Metrics::SearchPatterns(const vector &True, int value_A, const vector &B, int value_B)
  {
   int count=0;
   
   for(ulong i = 0; i < True.Size(); i++)
      if(True[i] == value_A && B[i] == value_B)
         count++;

   return count;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Metrics::rmse(const vector &True, const vector &Pred)
  {
   return Pred.RegressionMetric(True, REGRESSION_RMSE);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Metrics::mae(const vector &True, const vector &Pred)
  {
   return Pred.RegressionMetric(True, REGRESSION_MAE);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double Metrics::RegressionMetric(const vector &True,const vector &Pred,regression_metrics METRIC_)
 {
  double err = 0;
  
  switch (METRIC_)
   {
     case METRIC_MSE:
         err = mse(True, Pred);
         break;
     case METRIC_RMSE:
         err = rmse(True, Pred);
         break;
     case METRIC_MAE:
         err = mae(True, Pred);
         break;
     case METRIC_RSS:
         err = rss(True, Pred);
         break;
     case METRIC_R_SQUARED:
         err = r_squared(True, Pred);
         break;
     case METRIC_ADJUSTED_R:
         err = adjusted_r(True, Pred);
         break;
     default:
         break;
   }

  return err;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+