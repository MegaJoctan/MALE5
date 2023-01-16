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
class CMetrics
  {

protected:
   int SearchPatterns(vector &A, int value_A, vector &B, int value_B);
   
//-- From matrix utility class

   void VectorRemoveIndex(vector &v, ulong index);
   void MatrixRemoveRow(matrix &mat,ulong row); 
   void MatrixRemoveCol(matrix &mat, ulong col);
   
public:
                     CMetrics(void);
                    ~CMetrics(void);
                    
                    double r_squared(vector &A, vector &F); 
                    double adjusted_r(vector &A, vector &F,uint indep_vars=1);
//---
                
                   struct confusion_m
                    {
                        double accuracy;
                        vector precision;
                        vector recall;
                        vector f1_score;
                        vector specificity;
                        vector support;
                        
                        vector avg;
                        vector w_avg;
                        
                    } confusion_matrix_struct;
                    
                    double confusion_matrix(vector &A, vector &F,vector &classes, bool plot=true);
                    
                    double RSS(vector &A, vector &F);
                    double MSE(vector &A, vector &F);
                    
  };

//+------------------------------------------------------------------+

CMetrics::CMetrics(void)
  {
  
  }

//+------------------------------------------------------------------+

CMetrics::~CMetrics(void)
 {
 
 }

//+------------------------------------------------------------------+

double CMetrics::r_squared(vector &A,vector &P)
 {
   if (A.Size() != P.Size())
      {
         Print(__FUNCTION__," Vector A and P are not equal in size ");
         return(0);
      }
 
   double tss = 0, //total sum of squares
          rss;     //residual sum of squares
          
   vector c = MathPow(A-A.Mean(),2);
   
   tss = c.Sum();    
   
   c = MathPow(A-P,2);
   
   rss = c.Sum();
   
   return(1-(rss/tss));      
 }
 
//+------------------------------------------------------------------+

double CMetrics::adjusted_r(vector &A,vector &F,uint indep_vars=1)
 {
   if (A.Size() != F.Size())
      {
         Print(__FUNCTION__," Vector A and P are not equal in size ");
         return(0);
      }
      
   double r2 = r_squared(A,F);
   ulong N = F.Size();
   
   return(1-( (1-r2)*(N-1) )/(N - indep_vars -1));
 }
 
//+------------------------------------------------------------------+

double CMetrics::confusion_matrix(vector &A,vector &F,vector &classes,bool plot=true)
 {     
    ulong TP=0, TN=0, FP=0, FN=0;
     
    matrix conf_m(classes.Size(),classes.Size());
    conf_m.Fill(0); 
    
    vector row(classes.Size());       
    
    vector conf_v(ulong(MathPow(classes.Size(),2)));
    
    if (A.Size() != F.Size())
      {
         Print("A and F vectors are not same in size ");
         return(0);
      }

//---  
   
    for (ulong i=0; i<classes.Size(); i++)
      {  
       ulong col_=0, row_=0; 
         
//--- 
          for (ulong j=0; j<classes.Size(); j++)
             {                
                 conf_m[i][j] = SearchPatterns(A,(int)classes[i],F,(int)classes[j]);                  
             }
      }
      
//--- METRICS
   
   vector diag = conf_m.Diag();
   confusion_matrix_struct.accuracy = NormalizeDouble(diag.Sum()/conf_m.Sum(),3);

//--- precision
   
   confusion_matrix_struct.precision.Resize(classes.Size());
   vector col_v = {};
   
   for (ulong i=0; i<classes.Size(); i++)
      {
         col_v = conf_m.Col(i);
         VectorRemoveIndex(col_v,i);
         
         TP = (ulong)diag[i];
         FP = (ulong)col_v.Sum();
         
         confusion_matrix_struct.precision[i] = NormalizeDouble(TP/double(TP+FP),8);
      }

//--- recall

   vector row_v = {};
   confusion_matrix_struct.recall.Resize(classes.Size());
   
   for (ulong i=0; i<classes.Size(); i++)
      {
         row_v = conf_m.Row(i);
         VectorRemoveIndex(row_v,i);
         
         TP = (ulong)diag[i];
         FN = (ulong)row_v.Sum();
         
         confusion_matrix_struct.recall[i] = NormalizeDouble(TP/double(TP+FN),8);
      }

//--- specificity

   matrix temp_mat = {};
   ZeroMemory(col_v);
   
   confusion_matrix_struct.specificity.Resize(classes.Size());
   
   for (ulong i=0; i<classes.Size(); i++)
      {
          temp_mat.Copy(conf_m);
          
          MatrixRemoveCol(temp_mat,i);
          MatrixRemoveRow(temp_mat,i);
          
          col_v = conf_m.Col(i);
          VectorRemoveIndex(col_v,i);
         
          FP = (ulong)col_v.Sum();
          TN = (ulong)temp_mat.Sum(); 
          
          confusion_matrix_struct.specificity[i] = NormalizeDouble(TN/double(TN+FP),8);
      }

//--- f1 score

   confusion_matrix_struct.f1_score.Resize(classes.Size());
   
   for (ulong i=0; i<classes.Size(); i++)
     {
       confusion_matrix_struct.f1_score[i] = 2*((confusion_matrix_struct.precision[i]*confusion_matrix_struct.recall[i])/(confusion_matrix_struct.precision[i]+confusion_matrix_struct.recall[i]));      
       confusion_matrix_struct.f1_score[i] = NormalizeDouble(confusion_matrix_struct.f1_score[i],8);
     }

//--- support

   confusion_matrix_struct.support.Resize(classes.Size());
   
   ZeroMemory(row_v);
   for (ulong i=0; i<classes.Size(); i++)
     {
         row_v = conf_m.Row(i);
         confusion_matrix_struct.support[i] = row_v.Sum();
     }
     
   int total_size = (int)conf_m.Sum();
   
//--- Avg and w avg
   
   confusion_matrix_struct.avg.Resize(5);
   confusion_matrix_struct.w_avg.Resize(5);
     
    confusion_matrix_struct.avg[0] = confusion_matrix_struct.precision.Mean();       
    
    confusion_matrix_struct.avg[1] = confusion_matrix_struct.recall.Mean();
    confusion_matrix_struct.avg[2] = confusion_matrix_struct.specificity.Mean();
    confusion_matrix_struct.avg[3] = confusion_matrix_struct.f1_score.Mean();
    
    confusion_matrix_struct.avg[4] = total_size;
   
//--- w avg
    
   vector support_prop = confusion_matrix_struct.support/(double)total_size;
   
   vector c = confusion_matrix_struct.precision * support_prop;
   confusion_matrix_struct.w_avg[0] = c.Sum();
   
   c = confusion_matrix_struct.recall * support_prop;
   confusion_matrix_struct.w_avg[1] = c.Sum();
   
   c = confusion_matrix_struct.specificity * support_prop;
   confusion_matrix_struct.w_avg[2] = c.Sum();
   
   c = confusion_matrix_struct.f1_score * support_prop;
   confusion_matrix_struct.w_avg[3] = c.Sum();
   
   confusion_matrix_struct.w_avg[4] = (int)total_size;
   
//--- Report
   
   if (plot)
    {
      string report = "\n_\t\t\t\tPrecision \tRecall \tSpecificity \tF1 score \tSupport";
      
      for (ulong i=0; i<classes.Size(); i++)
         {
           report += "\n\t"+string(classes[i]);
             //for (ulong j=0; j<3; j++)
               report += StringFormat("\t\t\t %.2f \t\t\t %.2f \t\t\t %.2f \t\t\t %.2f \t\t\t %.1f",confusion_matrix_struct.precision[i],confusion_matrix_struct.recall[i],confusion_matrix_struct.specificity[i],confusion_matrix_struct.f1_score[i],confusion_matrix_struct.support[i]);
         }
     
      report += StringFormat("\n\nAccuracy\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t%.2f\n",confusion_matrix_struct.accuracy);
       
      report += StringFormat("Average \t %.2f \t\t %.2f \t\t %.2f \t\t %.2f \t\t %.1f",confusion_matrix_struct.avg[0],confusion_matrix_struct.avg[1],confusion_matrix_struct.avg[2],confusion_matrix_struct.avg[3],confusion_matrix_struct.avg[4]);
      report += StringFormat("\nW Avg \t\t\t %.2f \t\t %.2f \t\t %.2f \t\t %.2f \t\t %.1f",confusion_matrix_struct.w_avg[0],confusion_matrix_struct.w_avg[1],confusion_matrix_struct.w_avg[2],confusion_matrix_struct.w_avg[3],confusion_matrix_struct.w_avg[4]);
          
      Print("Confusion Matrix\n",conf_m);    
      Print("\nClassification Report\n",report);
   }
//---
  
      
   return (confusion_matrix_struct.accuracy);
 }

//+------------------------------------------------------------------+

double CMetrics::RSS(vector &A,vector &F)
 {
   vector c = A-F;
   c = MathPow(c,2);
   
   return (c.Sum()); 
 }

//+------------------------------------------------------------------+

double CMetrics::MSE(vector &A,vector &F)
 {
   vector c = A - F;
   c = MathPow(c,2);
   
   return(c.Sum()/c.Size()); 
 }

//+------------------------------------------------------------------+

int CMetrics::SearchPatterns(vector &A, int value_A, vector &B, int value_B)
 {
   int count=0;
  
   for (ulong i=0; i<A.Size(); i++)
     {
       if (A[i] == value_A && B[i] == value_B)
         { 
           count++;
         }
     }
      
    return count;
 }

//+------------------------------------------------------------------+

void CMetrics::VectorRemoveIndex(vector &v, ulong index)
  {
   vector new_v(v.Size()-1);

   for(ulong i=0, count = 0; i<v.Size(); i++)
      if(i != index)
        {
         new_v[count] = v[i];
         count++;
        }
    v.Copy(new_v);
  }

//+------------------------------------------------------------------+

void CMetrics::MatrixRemoveCol(matrix &mat, ulong col)
  {
   matrix new_matrix(mat.Rows(),mat.Cols()-1); //Remove the one Column

   for (ulong i=0, new_col=0; i<mat.Cols(); i++) 
     {
        if (i == col)
          continue;
        else
          {
           new_matrix.Col(mat.Col(i),new_col);
           new_col++;
          }    
     }

   mat.Copy(new_matrix);
  }
  
//+------------------------------------------------------------------+

void CMetrics::MatrixRemoveRow(matrix &mat,ulong row)
  {
   matrix new_matrix(mat.Rows()-1,mat.Cols()); //Remove the one Row
 
      for(ulong i=0, new_rows=0; i<mat.Rows(); i++)
        {
         if(i == row)
            continue;
         else
           {
            new_matrix.Row(mat.Row(i),new_rows);
            new_rows++;
           }
        }

   mat.Copy(new_matrix);
  }

//+------------------------------------------------------------------+

