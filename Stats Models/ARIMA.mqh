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

#include <MALE5\MatrixExtend.mqh>

class CARIMA
  {
public:

   vector difference(const vector &ts, uint interval=1);
   double inverse_difference(const vector &history, double y_hat, uint interval=1);
   vector auto_regressive(const vector &params, const vector &ts, uint p);
   vector moving_average(const vector &params, const vector &errors, uint q);
   
   double objective_function(const vector &params, const vector &ts, uint p, uint d, uint q); 
   
public:
                     CARIMA(void);
                    ~CARIMA(void);
                    
                    vector arima(const vector &params, const vector &ts, uint p, uint d, uint q);
                    vector gradient_descent(const vector &ts, uint p, uint d, uint q, double learning_rate=0.01, uint epochs=1000);
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
//|      Perform differencing to make time series stationary         |
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
   return y_hat + history[history.Size()-1];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//| The autoregressive function implements the autoregressive (AR)   |
//| component of the ARIMA model. The AR model predicts future values|   
//| in a time series as a linear combination of past values.         |
//|                                                                  |
//| params: The parameters of the AR model, typically the            |   
//|         coefficients for the past values.                        |            
//| ts: The time series data.                                        |
//| p: The order of the AR model, indicating how many past values    |
//|    to use for predicting the future value.                       |
//|                                                                  |      
//+------------------------------------------------------------------+
vector CARIMA::auto_regressive(const vector &params,const vector &ts,uint p)
 {
   vector ar_params = MatrixExtend::Slice(params, 0, p);
   vector ar_values = MatrixExtend::Zeros(ts.Size());
    
    for (uint t=p; t<ts.Size(); t++)
     {
       
       vector ts_slice = MatrixExtend::Slice(ts, t-p, t);
       MatrixExtend::Reverse(ts_slice);
       
       if (ts_slice.Size()<=0)
        {
          DebugBreak();
          break;
        }
       
       matrix ar_params_mat = MatrixExtend::VectorToMatrix(ar_params);
       matrix ts_slice_mat = MatrixExtend::VectorToMatrix(ts_slice, ts_slice.Size());
       
       ar_values[t] = ar_params_mat.MatMul(ts_slice_mat)[0][0]; //since the outcome is most likely a 1x1 matrix we conver it directly to double by accessing a value at index[0][0]
     }
   return ar_values;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//| The moving_average function implements the moving average (MA)   |
//| component of the ARIMA model. The MA model predicts future       |
//| values based on past error terms (residuals) rather than past    |
//| values of the time series itself.                                |
//|                                                                  |
//| Parameters:                                                      |
//|                                                                  |
//| params: The parameters of the ARIMA model, specifically the      |
//|          coefficients for the moving average part.               |
//| errors: The error terms (residuals) from the ARIMA model, which  |
//|         are used in the MA model.                                |
//| q: The order of the MA model, indicating how many past error     |
//|     terms to use for predicting the future value.                |
//|                                                                  |
//+------------------------------------------------------------------+
vector CARIMA::moving_average(const vector &params,const vector &errors,uint q)
 {
   vector ma_params = MatrixExtend::Slice(params, params.Size()-q, params.Size());
   vector ma_values = MatrixExtend::Zeros(errors.Size());
   
   for (uint t=q; t<errors.Size(); t++)
    {
      vector errors_slice = MatrixExtend::Slice(errors, t-q, t);
      MatrixExtend::Reverse(errors_slice);
      
      matrix ma_params_mat = MatrixExtend::VectorToMatrix(ma_params);
      matrix errors_params_mat = MatrixExtend::VectorToMatrix(errors_slice, errors_slice.Size()); 
      
      //Print("ma_params\n",ma_params_mat,"\nerror values\n",errors_params_mat);
      
      ma_values[t] = ma_params_mat.MatMul(errors_params_mat)[0][0];
      
      //Print("ma_values[",t,"]= ",ma_values);
    }
   return ma_values;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CARIMA::arima(const vector &params,const vector &ts,uint p,uint d,uint q)
 {
   vector diff_ts = difference(ts, d);
   vector ar_values = auto_regressive(params, diff_ts, p);
   vector residuals = diff_ts - ar_values;
   vector ma_values = moving_average(params, residuals, q);
   
   return ar_values + ma_values;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CARIMA::objective_function(const vector &params, const vector &ts, uint p, uint d, uint q)
 {
   vector arima_values = arima(params, ts, p, d, q);  
   vector diff_ts = difference(ts, d);
   
   vector residuals = diff_ts - arima_values;
   
   return MathPow(residuals.Sum(), 2); //SUM OF SQUARED RESIDUALS (SSR)
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CARIMA::gradient_descent(const vector &ts,uint p,uint d,uint q,double learning_rate=0.010000, uint epochs=1000)
 {
   uint min = 1, max = (uint)ts.Size();
   int random_state = MQLInfoInteger(MQL_DEBUG)?42:-1;
   
   vector params = MatrixExtend::Random(min, max, (int)p+q, random_state);
   
//---
   
   for (uint epoch=0; epoch<epochs; epoch++)
     {
       vector gradients = MatrixExtend::Zeros(params.Size());
        for (uint param=0; param<params.Size(); param++)
           {
              params[param] += 1e-5;
              double loss_1 = objective_function(params, ts, p, d, q);        
              params[param] -= 2 * 1e-5;               
              double loss_2 = objective_function(params, ts, p, d, q);
              
              gradients[param] = (loss_1 - loss_2) / (2 * 1e-5);
              params[param] += 1e-5; 
           }
           
        double loss = objective_function(params, ts, p, d, q);
        printf("epoch[%d/%d] loss = %.5f",epoch+1, epochs, loss);
        
        params -= learning_rate * gradients;
     }
     
   return params;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

