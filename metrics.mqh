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

struct confusion_matrix_struct
  {
   double            accuracy;
   vector<double>    precision;
   vector<double>    recall;
   vector<double>    f1_score;
   vector<double>    specificity;
   vector<double>    support;

   vector<double>    avg;
   vector<double>    w_avg;

  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CMetrics
  {
protected:
   static int SearchPatterns(vector &True, int value_A, vector &B, int value_B);

   //-- From matrix utility class

public:
   CMetrics(void);
   ~CMetrics(void);

   //--- Regression metrics

   static double r_squared(vector &True, vector &Pred);
   static double adjusted_r(vector &True, vector &Pred, uint indep_vars = 1);

   static double rss(vector &True, vector &Pred);
   static double mse(vector &True, vector &Pred);
   static double rmse(vector &True, vector &Pred);
   static double mae(vector &True, vector &Pred);

   //--- Classification metrics

   double accuracy_score(vector &True, vector &Pred);
   static confusion_matrix_struct confusion_matrix(vector &True, vector &Pred, bool report_show = true);
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
double CMetrics::r_squared(vector &True, vector &Pred)
  {
   return(Pred.RegressionMetric(True, REGRESSION_R2));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::adjusted_r(vector &True, vector &Pred, uint indep_vars = 1)
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
confusion_matrix_struct  CMetrics::confusion_matrix(vector &True, vector &Pred, bool report_show = true)
  {
   ulong TP = 0, TN = 0, FP = 0, FN = 0;

   vector classes = CMatrixutils::Unique(True);

   matrix conf_m(classes.Size(), classes.Size());
   conf_m.Fill(0);

   vector row(classes.Size());
   vector conf_v(ulong(MathPow(classes.Size(), 2)));


   confusion_matrix_struct confusion_mat;

   if(True.Size() != Pred.Size())
     {
      Print("True and Pred vectors are not same in size ");
      return confusion_mat;
     }

//---

   for(ulong i = 0; i < classes.Size(); i++)
     {
      ulong col_ = 0, row_ = 0;

      //---
      for(ulong j = 0; j < classes.Size(); j++)
        {
         conf_m[i][j] = SearchPatterns(True, (int)classes[i], Pred, (int)classes[j]);
        }
     }


   for(ulong i = 0; i < classes.Size(); i++)
     {
      ulong col_ = 0, row_ = 0;

      //---
      for(ulong j = 0; j < classes.Size(); j++)
        {
         conf_m[i][j] = SearchPatterns(True, (int)classes[i], Pred, (int)classes[j]);
        }
     }

//--- METRICS

   vector diag = conf_m.Diag();
   confusion_mat.accuracy = NormalizeDouble(diag.Sum() / conf_m.Sum(), 3);

//--- precision

   confusion_mat.precision.Resize(classes.Size());
   vector col_v = {};

   double value = 0;

   for(ulong i = 0; i < classes.Size(); i++)
     {
      col_v = conf_m.Col(i);
      CMatrixutils::VectorRemoveIndex(col_v, i);

      TP = (ulong)diag[i];
      FP = (ulong)col_v.Sum();

      value = TP / double(TP + FP);

      confusion_mat.precision[i] = NormalizeDouble(MathIsValidNumber(value) ? value : 0, 8);
     }

//--- recall

   vector row_v = {};
   confusion_mat.recall.Resize(classes.Size());

   for(ulong i = 0; i < classes.Size(); i++)
     {
      row_v = conf_m.Row(i);
      CMatrixutils::VectorRemoveIndex(row_v, i);

      TP = (ulong)diag[i];
      FN = (ulong)row_v.Sum();

      value = TP / double(TP + FN);

      confusion_mat.recall[i] = NormalizeDouble(MathIsValidNumber(value) ? value : 0, 8);
     }

//--- specificity

   matrix temp_mat = {};
   ZeroMemory(col_v);

   confusion_mat.specificity.Resize(classes.Size());

   for(ulong i = 0; i < classes.Size(); i++)
     {
      temp_mat.Copy(conf_m);

      CMatrixutils::RemoveCol(temp_mat, i);
      CMatrixutils::RemoveRow(temp_mat, i);

      col_v = conf_m.Col(i);
      CMatrixutils::VectorRemoveIndex(col_v, i);

      FP = (ulong)col_v.Sum();
      TN = (ulong)temp_mat.Sum();

      value = TN / double(TN + FP);

      confusion_mat.specificity[i] = NormalizeDouble(MathIsValidNumber(value) ? value : 0, 8);
     }

//--- f1 score

   confusion_mat.f1_score.Resize(classes.Size());

   for(ulong i = 0; i < classes.Size(); i++)
     {
      confusion_mat.f1_score[i] = 2 * ((confusion_mat.precision[i] * confusion_mat.recall[i]) / (confusion_mat.precision[i] + confusion_mat.recall[i]));

      value = confusion_mat.f1_score[i];

      confusion_mat.f1_score[i] = NormalizeDouble(MathIsValidNumber(value) ? value : 0, 8);
     }

//--- support

   confusion_mat.support.Resize(classes.Size());

   ZeroMemory(row_v);
   for(ulong i = 0; i < classes.Size(); i++)
     {
      row_v = conf_m.Row(i);
      confusion_mat.support[i] = NormalizeDouble(MathIsValidNumber(row_v.Sum()) ? row_v.Sum() : 0, 8);
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

   vector support_prop = confusion_mat.support / (double)total_size;

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

   if(report_show)
     {
      string report = "\n_\t\t\t\tPrecision \tRecall \tSpecificity \tF1 score \tSupport";

      for(ulong i = 0; i < classes.Size(); i++)
        {
         report += "\n\t" + string(classes[i]);
         //for (ulong j=0; j<3; j++)

         report += StringFormat("\t\t\t %.2f \t\t\t %.2f \t\t\t %.2f \t\t\t\t\t %.2f \t\t\t %.1f", confusion_mat.precision[i], confusion_mat.recall[i], confusion_mat.specificity[i], confusion_mat.f1_score[i], confusion_mat.support[i]);
        }

      report += StringFormat("\n\nAccuracy\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%.2f\n", confusion_mat.accuracy);

      report += StringFormat("Average \t %.2f \t\t %.2f \t\t %.2f \t\t\t\t %.2f \t\t %.1f", confusion_mat.avg[0], confusion_mat.avg[1], confusion_mat.avg[2], confusion_mat.avg[3], confusion_mat.avg[4]);
      report += StringFormat("\nW Avg \t\t\t %.2f \t\t %.2f \t\t %.2f \t\t\t\t %.2f \t\t %.1f", confusion_mat.w_avg[0], confusion_mat.w_avg[1], confusion_mat.w_avg[2], confusion_mat.w_avg[3], confusion_mat.w_avg[4]);

      Print("Confusion Matrix\n", conf_m);
      Print("\nClassification Report\n", report);
     }
//---

   return (confusion_mat);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::rss(vector &True, vector &Pred)
  {
   vector c = True - Pred;
   c = MathPow(c, 2);

   return (c.Sum());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::mse(vector &True, vector &Pred)
  {
   vector c = True - Pred;
   c = MathPow(c, 2);

   return(c.Sum() / c.Size());
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::accuracy_score(vector &True, vector &Pred)
  {
   return this.confusion_matrix(True, Pred, false).accuracy;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CMetrics::SearchPatterns(vector &True, int value_A, vector &B, int value_B)
  {
   int count = 0;

   for(ulong i = 0; i < True.Size(); i++)
     {
      if(True[i] == value_A && B[i] == value_B)
        {
         count++;
        }
     }

   return count;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CMetrics::rmse(vector &True, vector &Pred)
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