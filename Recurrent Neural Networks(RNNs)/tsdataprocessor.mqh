//+------------------------------------------------------------------+
//|                                                         RNNs.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\MatrixExtend.mqh>
#include <MALE5\Tensors.mqh>
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CTSDataProcessor 
  {
CTensors *tensor_memory[]; //Tensor objects may be hard to track in memory once we return them from a function, this keeps track of them
bool xandysplit;

public:
                     CTSDataProcessor (void);
                    ~CTSDataProcessor (void);
                    
                     CTensors *extract_timeseries_data(const matrix<double> &x, const int time_step); //for real time predictions
                     CTensors *extract_timeseries_data(const matrix<double> &MATRIX, vector &y, const int time_step); //for training and testing purposes 
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTSDataProcessor ::CTSDataProcessor (void)
 {
   xandysplit = true; //by default obtain the y vector also
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTSDataProcessor ::~CTSDataProcessor (void)
 {
   for (int i=0; i<ArraySize(tensor_memory); i++)
     if (CheckPointer(tensor_memory[i])!=POINTER_INVALID)
       delete(tensor_memory[i]);
 } 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensors *CTSDataProcessor ::extract_timeseries_data(const matrix<double> &x,const int time_step)
 {
  CTensors *timeseries_tensor;
  timeseries_tensor = new CTensors(0);
  ArrayResize(tensor_memory, 1);
  tensor_memory[0] = timeseries_tensor;
  
  xandysplit = false; //In this function we do not obtain the y vector
  
  vector y;
  timeseries_tensor = extract_timeseries_data(x, y, time_step);
  
  xandysplit = true; //restore the original condition
   
  return timeseries_tensor;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

CTensors *CTSDataProcessor ::extract_timeseries_data(const matrix &MATRIX, vector &y,const int time_step)
 {
  CTensors *timeseries_tensor;
  timeseries_tensor = new CTensors(0);
  ArrayResize(tensor_memory, 1);
  tensor_memory[0] = timeseries_tensor;
  
  matrix<double> time_series_data = {};
  matrix x = {}; //store the x variables converted to timeseries
  vector y_original = {};
  y.Init(0);
  
  if (xandysplit) //if we are required to obtain the y vector also split the given matrix into x and y
     if (!MatrixExtend::XandYSplitMatrices(MATRIX, x, y_original))
       {
         printf("%s failed to split the x and y matrices in order to make a tensor",__FUNCTION__);
         return timeseries_tensor;
       }
  
  x = xandysplit ? x : MATRIX; 
  
  for (ulong sample=0; sample<x.Rows(); sample++) //Go throught all the samples
    {
      matrix<double> time_series_matrix = {};
      vector<double> timeseries_y(1);
      
      for (ulong time_step_index=0; time_step_index<(ulong)time_step; time_step_index++)
        {
            if (sample + time_step_index >= x.Rows())
                break;
             
             time_series_matrix = MatrixExtend::concatenate(time_series_matrix, x.Row(sample+time_step_index), 0);
             
             if (xandysplit)
               timeseries_y[0] = y_original[sample+time_step_index]; //The last value in the column is assumed to be a y value so it gets added to the y vector
        }
      
      if (time_series_matrix.Rows()<(ulong)time_step)
        continue;
        
        //printf("[%d] size = (%dx%d)",sample,time_series_matrix.Rows(),time_series_matrix.Cols());
        timeseries_tensor.Append(time_series_matrix);
         
        if (xandysplit)
         y = MatrixExtend::concatenate(y, timeseries_y);
    }
   
   //Print("Timeseries Tensor");
   //timeseries_tensor.Print_();
   
   return timeseries_tensor;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
