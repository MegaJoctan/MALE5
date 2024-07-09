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
#include <MALE5\Neural Networks\optimizers.mqh>

class CARIMA
  {
protected:

   //vector roll(const vector &ts, int shift); 


   vector difference(const vector &ts, uint interval=1);
   double inverse_difference(const vector &history, double y_hat, uint interval=1);
   vector auto_regressive(const vector &params, const vector &ts, uint p);
   vector moving_average(const vector &params, const vector &errors, uint q);
   
   double objective_function(const vector &ts); 
   
//--- 
   
   //void adfuller_test(const vector &ts, uint max_lag=UINT_MAX);
   
   struct __config_struct__
     {
        uint p,d,q;
        vector params;
     }config;
   
public:
                     CARIMA(uint p, uint d, uint q);
                    ~CARIMA(void);
                    
                    vector arima(const vector &ts);
                    void fit(const vector &ts, OptimizerAdaGrad *optimizer, uint epochs=1000, bool verbose=true);
                    vector predict(const vector &ts);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CARIMA::CARIMA(uint p, uint d, uint q)
 {
   config.p = p;
   config.d = d;
   config.q = q;
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
   return y_hat + history[history.Size()-interval];
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
//| The arima function integrates the autoregressive (AR),           |
//| differencing (I), and moving average (MA) components to form the |
//| ARIMA model. This function is responsible for applying these     |
//| three components sequentially to model a time series.            |
//|                                                                  |
//| Parameters:                                                      |
//| params: The combined parameters for the AR and MA parts of the   |
//|         model.                                                   |
//| ts: The original time series data.                               |
//| p: The order of the AR component.                                |
//| d: The order of differencing to make the series stationary.      |
//| q: The order of the MA component                                 |
//|                                                                  |   
//+------------------------------------------------------------------+
vector CARIMA::arima(const vector &ts)
 {
   if (config.params.Size()==0)
     {
       printf("%s Error, Call the fit method first to train the model before using it for predictions",__FUNCTION__);
       DebugBreak();
       
       vector empty={};
       return empty;
     }
     
   vector diff_ts = difference(ts, config.d);
   vector ar_values = auto_regressive(config.params, diff_ts, config.p);
   vector residuals = diff_ts - ar_values;
   vector ma_values = moving_average(config.params, residuals, config.q);
   
   return ar_values + ma_values;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CARIMA::predict(const vector &ts)
 {
   vector y_hat = arima(ts);
   
   vector forecast(y_hat.Size());
   for (ulong i=0; i<y_hat.Size(); i++)
     forecast[i] = inverse_difference(ts, y_hat[i], config.d);
     
   return forecast;
 }
//+------------------------------------------------------------------+
//|      The function to optimize using gradient descent             |
//+------------------------------------------------------------------+
double CARIMA::objective_function(const vector &ts)
 {
   vector arima_values = arima(ts);  
   vector diff_ts = difference(ts, config.d);
   
   vector residuals = diff_ts - arima_values;
   
   return MathPow(residuals, 2).Sum(); //SUM OF SQUARED ERRORS (SSE)
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CARIMA::fit(const vector &ts, OptimizerAdaGrad *optimizer, uint epochs=1000, bool verbose=true)
 {
   uint min = 1, max = (uint)ts.Size();
   int random_state = MQLInfoInteger(MQL_DEBUG)?42:-1;
   
   config.params.Resize(int(config.p+config.q));
   config.params.Fill(0.0);
   
//---
   
   
   OptimizerAdaGrad adam = optimizer;
   
   for (uint epoch=0; epoch<epochs; epoch++)
     {
       vector gradients = MatrixExtend::Zeros(config.params.Size());
        for (uint param=0; param<config.params.Size(); param++)
           {
              config.params[param] += 1e-5;
              double loss_1 = objective_function(ts);        
              config.params[param] -= 2 * 1e-5;               
              double loss_2 = objective_function(ts);
              
              gradients[param] = (loss_1 - loss_2) / (2 * 1e-5);
              config.params[param] += 1e-5; 
           }
           
        double loss = objective_function(ts);
        
        if (verbose)
          printf("epoch[%d/%d] loss = %.5f",epoch+1, epochs, loss);
        
        matrix params_matrix = MatrixExtend::VectorToMatrix(config.params), 
               gradients_matrix = MatrixExtend::VectorToMatrix(gradients); //1D matrices
        
        adam.update(params_matrix, gradients_matrix);
        
        config.params = MatrixExtend::MatrixToVector(params_matrix);
     }
     
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
/*
void CARIMA::adfuller_test(const vector &ts, uint max_lag=UINT_MAX)
 {
   uint size = (uint)ts.Size();
   
   if (max_lag == UINT_MAX)
     max_lag = int(pow(size-1, 1/3)); //default max length 
   
     
//--- Calculate the test statistics
   
   vector diff = this.difference(ts);
   matrix lagged_diff = MatrixExtend::Zeros(max_lag, size-max_lag); //[1xn] matrix given the default max-lag
   
   for (int lag=1; lag<(int)max_lag+1; lag++)
     {
       vector rolled_vector = roll(diff, -lag);
       vector sliced_vector = MatrixExtend::Slice(rolled_vector,max_lag, rolled_vector.Size());
       
       lagged_diff.Row(sliced_vector, lag-1);
     }
     
   Print("lagged_diff\n",lagged_diff);
 }
*/
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
/*
vector CARIMA::roll(const vector &ts, int shift) 
{
   int n  = (int)ts.Size();
   vector temp_v(n);
   temp_v.Fill(0.0);
   
   // If shift is 0, do nothing
   if (shift == 0) 
     {
       printf("function=%s line=%d failed to roll a given vector",__FUNCTION__,__LINE__);
       return temp_v;
     }


   // Handle positive and negative shifts
   if (shift > 0) 
   {
      for (int i = 0; i < n; i++) 
      {
         temp_v[(i + shift) % n] = ts[i];
      }
   } 
   else 
   {
      shift = -shift;
      for (int i = 0; i < n; i++) 
      {
         temp_v[i] = ts[(i + shift) % n];
      }
   }
   return temp_v;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
