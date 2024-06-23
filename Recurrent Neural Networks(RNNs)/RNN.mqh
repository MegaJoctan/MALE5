//+------------------------------------------------------------------+
//|                                                          RNN.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

#include <MALE5\Tensors.mqh>
#define UNDEFINED_REPLACE 1

class CRNN
  {
protected:

   bool initialized;
   long onnx_handle;
   void PrintTypeInfo(const long num,const string layer,const OnnxTypeInfo& type_info);
   long inputs[], outputs[];
   
   void replace(long &arr[]) { for (uint i=0; i<arr.Size(); i++) if (arr[i] <= -1) arr[i] = UNDEFINED_REPLACE; }
   string ConvertTime(double seconds);
   
public:
                     CRNN(void);
                    ~CRNN(void);
                     
                     bool Init(const uchar &onnx_buff[], ulong flags=ONNX_DEFAULT);
                     bool Init(string onnx_filename, uint flags=ONNX_DEFAULT);

                     virtual int predict_bin(const matrix &x, const vector &classes_in_data);
                     virtual vector predict_bin(CTensors &timeseries_tensor, const vector &classes_in_data);
                     virtual vector predict_proba(const matrix &x);
                     
                     double predict(const matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRNN::CRNN(void):
   initialized(false)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRNN::~CRNN(void)
 {
   if (!OnnxRelease(onnx_handle))
     printf("%s Failed to release ONNX handle Err=%d",__FUNCTION__,GetLastError());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CRNN::Init(string onnx_filename, uint flags=ONNX_DEFAULT)
 {
   Print("Initilaizing ONNX model...");
  
   onnx_handle = OnnxCreate(onnx_filename, flags);
   
//--- since not all sizes defined in the input tensor we must set them explicitly
//--- first index - batch size, second index - series size, third index - number of series (only Close)
   
   OnnxTypeInfo type_info; //Getting onnx information for Reference In case you forgot what the loaded ONNX is all about

   long input_count=OnnxGetInputCount(onnx_handle);
   if (MQLInfoInteger(MQL_DEBUG))
      Print("model has ",input_count," input(s)");
   
   for(long i=0; i<input_count; i++)
     {
      string input_name=OnnxGetInputName(onnx_handle,i);
      if (MQLInfoInteger(MQL_DEBUG))
         Print(i," input name is ",input_name);
         
      if(OnnxGetInputTypeInfo(onnx_handle,i,type_info))
        {
          if (MQLInfoInteger(MQL_DEBUG))
            PrintTypeInfo(i,"input",type_info);
          ArrayCopy(inputs, type_info.tensor.dimensions);
        }
     }

   long output_count=OnnxGetOutputCount(onnx_handle);
   if (MQLInfoInteger(MQL_DEBUG))
      Print("model has ",output_count," output(s)");
   for(long i=0; i<output_count; i++)
     {
      string output_name=OnnxGetOutputName(onnx_handle,i);
      if (MQLInfoInteger(MQL_DEBUG))
         Print(i," output name is ",output_name);
         
      if(OnnxGetOutputTypeInfo(onnx_handle,i,type_info))
       {
         if (MQLInfoInteger(MQL_DEBUG))
            PrintTypeInfo(i,"output",type_info);
         ArrayCopy(outputs, type_info.tensor.dimensions);
       }
     }
   
//---
   
   replace(inputs);
   replace(outputs);
      
   if (!OnnxSetInputShape(onnx_handle, 0, inputs)) //Giving the Onnx handle the input shape
     {
       printf("Failed to set the input shape Err=%d",GetLastError());
       return false;
     }
   
   if (!OnnxSetOutputShape(onnx_handle, 0, outputs)) //giving the onnx handle the output shape
     {
       printf("Failed to set the Output shape Err=%d",GetLastError());
       return false;
     } 
   
   initialized = true;
   Print("ONNX model Initialized");
   
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CRNN::Init(const uchar &onnx_buff[], ulong flags=ONNX_DEFAULT)
 {
  Print("Initilaizing ONNX model..."); 
  
  onnx_handle = OnnxCreateFromBuffer(onnx_buff, flags); //creating onnx handle buffer | rUN DEGUG MODE during debug mode
  
  
//--- since not all sizes defined in the input tensor we must set them explicitly
//--- first index - batch size, second index - series size, third index - number of series (only Close)
   
   OnnxTypeInfo type_info; //Getting onnx information for Reference In case you forgot what the loaded ONNX is all about

   long input_count=OnnxGetInputCount(onnx_handle);
   if (MQLInfoInteger(MQL_DEBUG))
      Print("model has ",input_count," input(s)");
   
   for(long i=0; i<input_count; i++)
     {
      string input_name=OnnxGetInputName(onnx_handle,i);
      if (MQLInfoInteger(MQL_DEBUG))
         Print(i," input name is ",input_name);
         
      if(OnnxGetInputTypeInfo(onnx_handle,i,type_info))
        {
          if (MQLInfoInteger(MQL_DEBUG))
            PrintTypeInfo(i,"input",type_info);
          ArrayCopy(inputs, type_info.tensor.dimensions);
        }
     }

   long output_count=OnnxGetOutputCount(onnx_handle);
   if (MQLInfoInteger(MQL_DEBUG))
      Print("model has ",output_count," output(s)");
   for(long i=0; i<output_count; i++)
     {
      string output_name=OnnxGetOutputName(onnx_handle,i);
      if (MQLInfoInteger(MQL_DEBUG))
         Print(i," output name is ",output_name);
         
      if(OnnxGetOutputTypeInfo(onnx_handle,i,type_info))
       {
         if (MQLInfoInteger(MQL_DEBUG))
            PrintTypeInfo(i,"output",type_info);
         ArrayCopy(outputs, type_info.tensor.dimensions);
       }
     }
   
//---
   
   replace(inputs);
   replace(outputs);
      
   if (!OnnxSetInputShape(onnx_handle, 0, inputs)) //Giving the Onnx handle the input shape
     {
       printf("Failed to set the input shape Err=%d",GetLastError());
       return false;
     }
   
   if (!OnnxSetOutputShape(onnx_handle, 0, outputs)) //giving the onnx handle the output shape
     {
       printf("Failed to set the Output shape Err=%d",GetLastError());
       return false;
     } 
   
   initialized = true;
   
   Print("ONNX model Initialized");
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CRNN::PrintTypeInfo(const long num,const string layer,const OnnxTypeInfo& type_info)
  {
   Print("   type ",EnumToString(type_info.type));
   Print("   data type ",EnumToString(type_info.type));

   if(type_info.tensor.dimensions.Size()>0)
     {
      bool   dim_defined=(type_info.tensor.dimensions[0]>0);
      string dimensions=IntegerToString(type_info.tensor.dimensions[0]);
      
      
      for(long n=1; n<type_info.tensor.dimensions.Size(); n++)
        {
         if(type_info.tensor.dimensions[n]<=0)
            dim_defined=false;
         dimensions+=", ";
         dimensions+=IntegerToString(type_info.tensor.dimensions[n]);
        }
      Print("   shape [",dimensions,"]");
      //--- not all dimensions defined
      if(!dim_defined)
         PrintFormat("   %I64d %s shape must be defined explicitly before model inference",num,layer);
      //--- reduce shape
      uint reduced=0;
      long dims[];
      for(long n=0; n<type_info.tensor.dimensions.Size(); n++)
        {
         long dimension=type_info.tensor.dimensions[n];
         //--- replace undefined dimension
         if(dimension<=0)
            dimension=UNDEFINED_REPLACE;
         //--- 1 can be reduced
         if(dimension>1)
           {
            ArrayResize(dims,reduced+1);
            dims[reduced++]=dimension;
           }
        }
      //--- all dimensions assumed 1
      if(reduced==0)
        {
         ArrayResize(dims,1);
         dims[reduced++]=1;
        }
      //--- shape was reduced
      if(reduced<type_info.tensor.dimensions.Size())
        {
         dimensions=IntegerToString(dims[0]);
         for(long n=1; n<dims.Size(); n++)
           {
            dimensions+=", ";
            dimensions+=IntegerToString(dims[n]);
           }
         string sentence="";
         if(!dim_defined)
            sentence=" if undefined dimension set to "+(string)UNDEFINED_REPLACE;
         PrintFormat("   shape of %s data can be reduced to [%s]%s",layer,dimensions,sentence);
        }
     }
   else
      PrintFormat("no dimensions defined for %I64d %s",num,layer);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CRNN::predict_bin(const matrix &x, const vector &classes_in_data)
 {
   if (!this.initialized)
    {
      printf("%s The model is not initialized yet to make predictions | call Init function first",__FUNCTION__);
      return 0;
    }

//---
   
   matrixf x_float;
   x_float.Assign(x);
   
   vector output_data(this.outputs[outputs.Size()-1]);
   
   if (!OnnxRun(onnx_handle, ONNX_DATA_TYPE_FLOAT, x_float, output_data))
     {
       printf("Failed to get predictions from Onnx err %d",GetLastError());
       return false;
     }
     
   return (int)classes_in_data[output_data.ArgMax()]; //Return the class with highest probability
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CRNN::predict_proba(const matrix &x)
 {
 
   if (!this.initialized)
    {
      printf("%s The model is not initialized yet to make predictions | call Init function first",__FUNCTION__);
      vector empty = {};
      return empty;
    }

//---
   
   matrixf x_float;
   x_float.Assign(x);
   
   vector proba(this.outputs[outputs.Size()-1]);
   
   if (!OnnxRun(onnx_handle, ONNX_DATA_TYPE_FLOAT, x_float, proba))
     {
       printf("Failed to get predictions from Onnx err %d",GetLastError());
       return proba;
     }
     
   return proba; //Return the class with highest probability
 }
//+------------------------------------------------------------------+
//|  When given a matrix for timeseries data collected it provides   |
//| a scalar binary value which is a prediction                      |
//+------------------------------------------------------------------+
vector CRNN::predict_bin(CTensors &timeseries_tensor, const vector &classes_in_data)
 {
   vector preds(timeseries_tensor.SIZE);
   for (uint i=0; i<timeseries_tensor.SIZE; i++)
    {
      matrix x = timeseries_tensor.Get(i);
      preds[i] = predict_bin(x, classes_in_data);
    }
      
   return preds;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CRNN::predict(const matrix &x)
 {
   if (!this.initialized)
    {
      printf("%s The model is not initialized yet to make predictions | call Init function first",__FUNCTION__);
      return 0;
    }

//---
   
   matrixf x_float;
   x_float.Assign(x);
   
   vector output_data(this.outputs[outputs.Size()-1]);
   
   if (!OnnxRun(onnx_handle, ONNX_DATA_TYPE_FLOAT, x_float, output_data))
     {
       printf("Failed to get predictions from Onnx err %d",GetLastError());
       return false;
     }
     
   return output_data[0]; //Return the class with highest probability
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CRNN::ConvertTime(double seconds)
{
    string time_str = "";
    uint minutes = 0, hours = 0;

    if (seconds >= 60)
    {
        minutes = (uint)(seconds / 60.0) ;
        seconds = fmod(seconds, 1.0) * 60;
        time_str = StringFormat("%d Minutes and %.3f Seconds", minutes, seconds);
    }
    
    if (minutes >= 60)
    {
        hours = (uint)(minutes / 60.0);
        minutes = minutes % 60;
        time_str = StringFormat("%d Hours and %d Minutes", hours, minutes);
    }

    if (time_str == "")
    {
        time_str = StringFormat("%.3f Seconds", seconds);
    }

    return time_str;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
