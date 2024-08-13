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
   vector residuals; //The differences between actual and predicted values.
   vector theta; //The coefficients of the AR model.
   double intercept; //The intercept term from the regression model.
 };

struct ma_struct
 {
   vector theta;
   double intercept;
 };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
 
class CARIMA
  {
protected:

   CLinearRegression  *ar_lr, *ma_lr;
   
   
   vector Shift(const vector &v, int shift); 
   vector Pad(const vector &v, int padSize, double padValue=EMPTY_VALUE);
   
   ar_struct AR(const uint p, const vector &series_data);
   ma_struct MA(const uint q, const vector &residuals);
   
   uint __p__,__d__, __q__;  
    
   ar_struct ar_parameters;
   ma_struct ma_parameters;
                     
public:
                     CARIMA(const uint p, const uint d, const uint q);
                    ~CARIMA(void);
                    
                     vector difference(const vector &ts, uint interval=1);
                     double inverse_difference(const vector &history, double y_hat, uint interval=1);
                    
                     void fit(const vector &series);
                     vector predict(const vector &series, const uint steps=10);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CARIMA::CARIMA(const uint p, const uint d, const uint q)
 :__p__(p),
  __q__(q),
  __d__(d)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CARIMA::~CARIMA(void)
 {
   if (CheckPointer(ar_lr) != POINTER_INVALID)
     delete ar_lr;
     
   if (CheckPointer(ma_lr) != POINTER_INVALID)
     delete ma_lr;
 }
//+------------------------------------------------------------------+
//|   This function shifts series_data similarly to                  |
//|   pandas.DataFrame.shift                                         |
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
//|      Function to pad a vector with a specified value             |
//+------------------------------------------------------------------+
vector CARIMA::Pad(const vector &v, int padSize, double padValue=EMPTY_VALUE)
{
   int originalSize = (int)v.Size();
   int newSize = originalSize + padSize;
   
   vector results(newSize); //a vector to hold the padded results
   
   // Fill the beginning of the array with the padValue
   for (int i = 0; i < padSize; i++)
      results[i] = padValue;
   
   // Copy the original array to the new array after the padding
   for (int i = 0; i < originalSize; i++)
      results[i + padSize] = v[i];
   
   return results;
}
//+------------------------------------------------------------------+
//|                                                                  |
//|      Perform differencing to make time series stationary         |
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
//|  To invert the differencing, we need to add the differenced value|
//|  y_hat back to the last observed value before the differencing.  |
//|                                                                  |
//|  Parameters                                                      |     
//|  history: The original time series data before differencing.     |
//|  yhat: The differenced value or forecast that we want to convert |
//|         back to the original scale.                              |
//|  interval: The differencing interval, default value is 1 for     |
//|            first-order differencing                              |
//|                                                                  |
//+------------------------------------------------------------------+
double CARIMA::inverse_difference(const vector &history, double y_hat, uint interval=1)
 {
   return y_hat + history[history.Size()-interval];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//|  This function fits an AutoRegressive model of order p to the    |
//|  time series data by leveraging linear regression.               |
//|                                                                  |
//|  It predicts the current value of the time series based on its   |
//|  previous p values, calculates the prediction errors, and        |
//|  provides the arima model coefficients and intercept values.     |
//|                                                                  |   
//|                                                                  |
//|  p: The order of the AR model, which specifies how many lagged   |
//|     values of the time series to include as predictors.          |
//|  series_data: The time series series_data.                       |
//|                                                                  |
//+------------------------------------------------------------------+
ar_struct CARIMA::AR(const uint p, const vector &series_data)
 {
   ar_struct ret_struct;
   
   ulong size = series_data.Size();
   
   matrix autoregressive_series_data(size, p+1);
   autoregressive_series_data.Col(series_data, 0);
   
   vector shifted_series_data = {};
 
    for (ulong i=1; i<p+1; i++) //generate lagged values
      {
         shifted_series_data = Shift(series_data, (uint)i);
         autoregressive_series_data.Col(shifted_series_data, i);
      } 
      
//---
      
   autoregressive_series_data = MatrixExtend::Slice(autoregressive_series_data,p,-1); 
   
//--- Since the y vector is the first column of this matrix
   
   matrix X;
   vector y;
   
   MatrixExtend::XandYSplitMatrices(autoregressive_series_data, X, y, 0); //index of 0 gets the first column assigned as y vector
   
//--- Fitting a linear regression model to the outo-regressive series_data
   
   ar_lr = new CLinearRegression();
   
   ar_lr.fit(X, y);
   vector y_pred = ar_lr.predict(X);
      
//---

   ret_struct.residuals = y - y_pred;
   ret_struct.theta = ar_lr.coeff_;
   ret_struct.intercept = ar_lr.intercept_;
   
   if (MQLInfoInteger(MQL_DEBUG))
     printf("AR(p=%d) model - RMSE: %.4f",p,y_pred.RegressionMetric(y, REGRESSION_RMSE));
   
   return ret_struct;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//|  The MA function constructs a Moving Average (MA) model by       |
//|  regressing the current residuals against the past q residuals.  |
//|  It uses linear regression to determine the coefficients theta   |
//|  and the intercept that best fit the relationship between the    |
//|  residuals and their lagged values.                              |   
//|  The resulting model helps in forecasting future residuals based |   
//|  on past errors.                                                 |
//|                                                                  |
//|  Parameters:                                                     |
//|  q: The order of the MA model, indicating how many lagged        |
//|     residuals (errors) should be considered.                     |
//|  residuals: The array of residuals (errors) from the previous AR |
//|     model fitting.                                               |
//|                                                                  |
//+------------------------------------------------------------------+
ma_struct CARIMA::MA(const uint q,const vector &residuals)
 {
   ma_struct ret_struct;
   
   ulong size = residuals.Size();
   
   matrix autoregressive_residuals(size, q+1);
   autoregressive_residuals.Col(residuals, 0);
   
   vector shifted_eesiduals = {};
 
    for (ulong i=1; i<q+1; i++) //This loop creates lagged versions of the residuals.
      {
         shifted_eesiduals = Shift(residuals, (uint)i);
         autoregressive_residuals.Col(shifted_eesiduals, i);
      } 
     
//---
      
   autoregressive_residuals = MatrixExtend::Slice(autoregressive_residuals,q,-1); 
   
//--- Since the y vector is the first column of this matrix
   
   matrix X;
   vector y;
   
   MatrixExtend::XandYSplitMatrices(autoregressive_residuals, X, y, 0); //index of 0 gets the first column assigned as y vector
   
//--- Fitting a linear regression model to the outo-regressive residuals
   
   ma_lr = new CLinearRegression();
   ma_lr.fit(X, y);
   vector y_pred = ma_lr.predict(X);
      
//---
   
   ret_struct.theta = ma_lr.coeff_;
   ret_struct.intercept = ma_lr.intercept_;   

//---

   if (MQLInfoInteger(MQL_DEBUG))
     printf("MA(q=%d) model - RMSE: %.4f",q,y_pred.RegressionMetric(y, REGRESSION_RMSE));
     
   return ret_struct;   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CARIMA::fit(const vector &series)
 {
   ar_parameters = AR(__p__, series);
   ma_parameters = MA(__q__, ar_parameters.residuals); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CARIMA::predict(const vector &series,const uint steps=10)
 {
    vector forecasted_values(steps);
    vector temp_series = series;
    
    //--- We initialize the residuals for the new data
    
    vector temp_residuals(series.Size() + steps);
    temp_residuals.Fill(0);
    
    for (uint step=0; step<steps; step++)
      {
        
        //--- Auto-regressive part
        
           vector ar_terms = MatrixExtend::Slice(temp_series,temp_series.Size()-__p__, -1);
           MatrixExtend::Reverse(ar_terms);
           
           double ar_part = ar_parameters.theta.MatMul(ar_terms) + ar_parameters.intercept;
           
        //--- Moving-average part | Generating MA terms
        
           vector ma_terms = MatrixExtend::Slice(temp_residuals,temp_residuals.Size()-__q__, -1);
           MatrixExtend::Reverse(ma_terms);
           
           double ma_part = ma_parameters.theta.MatMul(ma_terms) + ma_parameters.intercept;
           
        //--- Calculate forecast value difference 
        
           double forecast_value_diff = ar_part + ma_part;
           double new_value = temp_series[temp_series.Size()-1] + forecast_value_diff; // We use the last value in the series to convert the differenced forecast back to original scale

           forecasted_values[step] = new_value;
           
         //--- Update the temp_residuals with the new forecasted value
           
           temp_series = MatrixExtend::concatenate(temp_series, new_value);
           
           Print("Temp series\n",temp_series);
           
           double new_residual = forecast_value_diff;
           temp_residuals = MatrixExtend::concatenate(temp_residuals, new_residual);
           
           Print("Temp residuals\n",temp_residuals);
      }
          
    return forecasted_values;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
