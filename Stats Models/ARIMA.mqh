//+------------------------------------------------------------------+
//|                                                        ARIMA.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

#include <MALE5\Linear Models\Linear Regression.mqh>
#include <MALE5\MatrixExtend.mqh>

struct ar_struct
 {
   vector residuals;
   vector theta;
   double intercept;
 };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
 
class CARIMA
  {
protected:

   CLinearRegression  lr;
   vector difference(const vector &ts, uint interval=1);
   vector Shift(const vector &v, int shift); 

public:
                     CARIMA(void);
                    ~CARIMA(void);
                    
                     ar_struct AR(const uint p, const vector &data);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CARIMA::CARIMA(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CARIMA::~CARIMA(void)
 {
 
 }
//+------------------------------------------------------------------+
//|   This function shifts data similarly to pandas.DataFrame.shift  |
//+------------------------------------------------------------------+
vector CARIMA::Shift(const vector &v, int shift) 
 {
   int size = (int)v.Size();
   
   vector new_v(v.Size());
   new_v.Fill(EMPTY_VALUE);
   
   // If shift is positive, shift right
   if(shift > 0) 
    {
      for(int i = size - 1; i >= shift; i--) 
        new_v[i] = v[i - shift];
     }
   
   // If shift is negative, shift left
   else if(shift < 0) 
    {
      shift = -shift;
      for(int i = 0; i < size - shift; i++) 
        new_v[i] = v[i + shift];
     }
     
   return new_v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CARIMA::difference(const vector &ts, uint interval=1)
 {
   if (interval>=ts.Size())
     {
       printf("%s fatal, interval=%d must be less than the timeseries vector size=%d",__FUNCTION__,interval,ts.Size());
       vector empty={};
       return empty;
     }
   
   vector diff(ts.Size()-interval);
   
   for (uint i=interval, count=0; i<ts.Size(); i++)
     diff[i-interval] = ts[i] - ts[i-interval];
     
   return diff;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ar_struct CARIMA::AR(const uint p, const vector &data)
 {
   ar_struct ret_struct;
   
   ulong size = data.Size();
   
   matrix autoregressive_data(size, p+1);
   autoregressive_data.Col(data, 0);
   
   vector shifted_data = {};
 
    for (ulong i=1; i<p+1; i++)
      {
         shifted_data = Shift(data, (uint)i);
         autoregressive_data.Col(shifted_data, i);
      } 
      
//---
      
   autoregressive_data = MatrixExtend::Slice(autoregressive_data,p,-1); 
   
//--- Since the y vector is the first column of this matrix
   
   matrix X;
   vector y;
   
   MatrixExtend::XandYSplitMatrices(autoregressive_data, X, y, 0); //index 0 to get the first column assigned as y vector
   
//--- Fitting a linear regression model to the outo-regressive data
   
   lr.fit(X, y);
   vector y_pred = lr.predict(X);
      
//---

   ret_struct.residuals = y - y_pred;
   ret_struct.theta = lr.coeff_;
   ret_struct.intercept = lr.intercept_;
   
   if (MQLInfoInteger(MQL_DEBUG))
     printf("AR(p=%d) model - RMSE: %.4f",p,y_pred.RegressionMetric(y, REGRESSION_RMSE));
   
   return ret_struct;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
