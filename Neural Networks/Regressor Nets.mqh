//+------------------------------------------------------------------+
//|                                                neural_nn_lib.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
//|  Regressor Neural Networks | Neural Networks for solving         |
//|  regression problems in contrast to classification problems,     |
//|  here we deal with continuous variables                          |
//+------------------------------------------------------------------+

#include <MALE5\preprocessing.mqh>;
#include <MALE5\MatrixExtend.mqh>;
#include <MALE5\Metrics.mqh>
#include <MALE5\Tensors.mqh>
#include <MALE5\cross_validation.mqh>
#include "optimizers.mqh"


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

enum activation
  {
   AF_ELU_ = AF_ELU,
   AF_EXP_ = AF_EXP,
   AF_GELU_ = AF_GELU,
   AF_LINEAR_ = AF_LINEAR,
   AF_LRELU_ = AF_LRELU,
   AF_RELU_ = AF_RELU,
   AF_SELU_ = AF_SELU,
   AF_TRELU_ = AF_TRELU,
   AF_SOFTPLUS_ = AF_SOFTPLUS
  };
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

enum loss
  {
    LOSS_MSE_ = LOSS_MSE,  // Mean Squared Error
    LOSS_MAE_ = LOSS_MAE,  // Mean Absolute Error
    LOSS_MSLE_ = LOSS_MSLE,  // Mean Squared Logarithmic Error
    LOSS_POISSON_ = LOSS_POISSON  // Poisson Loss
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

struct backprop //This structure returns the loss information obtained from the backpropagation function
  {
    vector training_loss,
           validation_loss;
           
           void Init(ulong epochs)
            {
              training_loss.Resize(epochs);
              validation_loss.Resize(epochs);
            }
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

struct mlp_struct //multi layer perceptron information structure
 {
   ulong inputs;
   ulong hidden_layers;
   ulong outputs;
 };
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CRegressorNets
  {   
   mlp_struct        mlp;
   
   CTensors          *Weights_tensor; //Weight Tensor
   CTensors          *Bias_tensor;
   CTensors          *Input_tensor;
   CTensors          *Output_tensor;
   
protected:
   activation        A_FX;
   loss              m_loss_function;
   bool              trained;

   string            ConvertTime(double seconds);

//-- for backpropn

   vector            W_CONFIG;
   vector            HL_CONFIG; 
   
   bool              isBackProp; 
   matrix<double>    ACTIVATIONS;
   matrix<double>    Partial_Derivatives; 
   int               m_random_state;
   
private: 
   
   virtual backprop  backpropagation(const matrix& x, const vector &y, OptimizerSGD *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual backprop  backpropagation(const matrix& x, const vector &y, OptimizerAdaDelta *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual backprop  backpropagation(const matrix& x, const vector &y, OptimizerAdaGrad *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual backprop  backpropagation(const matrix& x, const vector &y, OptimizerAdam *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual backprop  backpropagation(const matrix& x, const vector &y, OptimizerNadam *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual backprop  backpropagation(const matrix& x, const vector &y, OptimizerRMSprop *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   
public:

                              CRegressorNets(vector &HL_NODES, activation AF_=AF_RELU_, loss m_loss_function=LOSS_MSE_, int random_state=42);
                             ~CRegressorNets(void);

   virtual void              fit(const matrix &x, const vector &y, OptimizerSGD *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual void              fit(const matrix &x, const vector &y, OptimizerAdaDelta *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual void              fit(const matrix &x, const vector &y, OptimizerAdaGrad *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual void              fit(const matrix &x, const vector &y, OptimizerAdam *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual void              fit(const matrix &x, const vector &y, OptimizerNadam *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   virtual void              fit(const matrix &x, const vector &y, OptimizerRMSprop *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false);
   
   virtual double            predict(const vector &x);
   virtual vector            predict(const matrix &x);
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRegressorNets::CRegressorNets(vector &HL_NODES, activation AF_=AF_RELU_, loss LOSS_=LOSS_MSE_, int random_state=42)
 :A_FX(AF_),
  m_loss_function(LOSS_),
  isBackProp(false),
  m_random_state(random_state)
  {   
     HL_CONFIG.Copy(HL_NODES);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRegressorNets::~CRegressorNets(void)
  {
   if (CheckPointer(this.Weights_tensor) != POINTER_INVALID)  delete(this.Weights_tensor);
   if (CheckPointer(this.Bias_tensor) != POINTER_INVALID)  delete(this.Bias_tensor);
   if (CheckPointer(this.Input_tensor) != POINTER_INVALID)  delete(this.Input_tensor);
   if (CheckPointer(this.Output_tensor) != POINTER_INVALID)  delete(this.Output_tensor);
   
   isBackProp = false; 
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CRegressorNets::predict(const vector &x)
  {   
  
  if (!trained)
   {
     printf("%s Train the model first before using it to make predictions | call the fit function first",__FUNCTION__);
     return 0;
   }
   
   matrix L_INPUT = MatrixExtend::VectorToMatrix(x); 
   matrix L_OUTPUT ={};
    
   for(ulong i=0; i<mlp.hidden_layers; i++)
     {      
      if (isBackProp) //if we are on backpropagation store the inputs to be used for finding derivatives 
        this.Input_tensor.Add(L_INPUT, i);  

      L_OUTPUT = this.Weights_tensor.Get(i).MatMul(L_INPUT) + this.Bias_tensor.Get(i); //Weights x INputs + Bias 

      L_OUTPUT.Activation(L_OUTPUT, ENUM_ACTIVATION_FUNCTION(A_FX)); //Activation
      
      L_INPUT = L_OUTPUT; //Next layer inputs = previous layer outputs
      
      if (isBackProp)  this.Output_tensor.Add(L_OUTPUT, i); //Add bias //if we are on backpropagation store the outputs to be used for finding derivatives 
     }
   
   return(L_OUTPUT[0][0]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CRegressorNets::predict(const matrix &x)
 {
  ulong size = x.Rows();
  
  vector v(size);
   if (x.Cols() != mlp.inputs)
    {
       Print("Cen't pass this matrix to a MLP it doesn't have the same number of columns as the inputs given primarily");
       return (v); 
    }

   for (ulong i=0; i<size; i++)
     v[i] = predict(x.Row(i));    
    
   return (v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

string CRegressorNets::ConvertTime(double seconds)
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
backprop CRegressorNets::backpropagation(const matrix& x, const vector &y, OptimizerSGD *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
   isBackProp = true;
   
//---

   backprop backprop_struct;
   backprop_struct.Init(epochs);
   
   ulong rows = x.Rows();
   
   mlp.inputs = x.Cols();
   mlp.outputs = 1;
   
//---

   vector v2 = {(double)mlp.outputs}; //Adding the output layer to the mix of hidden layers
  
   HL_CONFIG = MatrixExtend::concatenate(HL_CONFIG, v2);
   mlp.hidden_layers = HL_CONFIG.Size();
   W_CONFIG.Resize(HL_CONFIG.Size());
     
//---

   if (y.Size() != rows)
     {
        Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
        return backprop_struct;
     }
     
     
     matrix W, B;
     
//--- GENERATE WEIGHTS
    
     this.Weights_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Bias_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Input_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Output_tensor = new CTensors((uint)mlp.hidden_layers);
     
     ulong layer_input = mlp.inputs; 
     
     for (ulong i=0; i<mlp.hidden_layers; i++)
       {
          W_CONFIG[i] = layer_input*HL_CONFIG[i];
          
          W = MatrixExtend::Random(0.0, 1.0,(ulong)HL_CONFIG[i],layer_input, m_random_state);
          
          W = W * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Weights_tensor.Add(W, i);
          
          B = MatrixExtend::Random(0.0, 0.5,(ulong)HL_CONFIG[i],1,m_random_state);
          
          B = B * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Bias_tensor.Add(B, i);
          
          layer_input = (ulong)HL_CONFIG[i];
       }
     
//---

   if (MQLInfoInteger(MQL_DEBUG))
      Comment("<-------------------  R E G R E S S O R   N E T S  ------------------------->\n",
            "HL_CONFIG ",HL_CONFIG," TOTAL HL(S) ",mlp.hidden_layers,"\n",
            "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
            "NN INPUTS ",mlp.inputs," OUTPUT ",mlp.outputs
           );

//--- Optimizer
      
   OptimizerSGD optimizer_weights = optimizer;
   OptimizerSGD optimizer_bias = optimizer;
   
   if (batch_size>0)
    {
      OptimizerMinBGD optimizer_weights;
      OptimizerMinBGD optimizer_bias;
    }
     
//--- Cross validation

    CCrossValidation cross_validation;      
    CTensors *cv_tensor;
    matrix validation_data = MatrixExtend::concatenate(x, y);
    matrix validation_x;
    vector validation_y;
    
    cv_tensor = cross_validation.KFoldCV(validation_data, 10); //k-fold cross validation | 10 folds selected
    
//---

    matrix DELTA = {};
    double actual=0, pred=0;
    
    matrix temp_inputs ={};
    
    matrix dB = {}; //Bias Derivatives
    matrix dW = {}; //Weight Derivatives
    
   
    for (ulong epoch=0; epoch<epochs && !IsStopped(); epoch++)
      {        
        double epoch_start = GetTickCount(); 

        uint num_batches = (uint)MathFloor(x.Rows()/(batch_size+DBL_EPSILON));
        
        vector batch_loss(num_batches), 
               batch_accuracy(num_batches);
                       
         vector actual_v(1), pred_v(1), LossGradient = {};
         
         if (batch_size==0) //Stochastic Gradient Descent
          {
           for (ulong iter=0; iter<rows; iter++) //iterate through all data points
             {
               pred = predict(x.Row(iter));
               actual = y[iter];
               
               pred_v[0] = pred; 
               actual_v[0] = actual; 
   //---
                
                DELTA.Resize(mlp.outputs,1);
                
                for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                  {    
                     Partial_Derivatives = this.Output_tensor.Get(int(layer));
                     temp_inputs = this.Input_tensor.Get(int(layer));
                     
                     Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                     
                     if (mlp.hidden_layers-1 == layer) //Last layer
                      {                     
                        LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                        
                        DELTA.Col(LossGradient, 0);
                      }
                      
                     else
                      {
                        W = this.Weights_tensor.Get(layer+1);
                        
                        DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                      }
                    
                    //-- Observation | DeLTA matrix is same size as the bias matrix
                    
                    W = this.Weights_tensor.Get(layer);
                    B = this.Bias_tensor.Get(layer);
                  
                   //--- Derivatives wrt weights and bias
                  
                    dB = DELTA;
                    dW = DELTA.MatMul(temp_inputs.Transpose());                   
                    
                   //--- Weights updates
                    
                    optimizer_weights.update(W, dW);
                    optimizer_bias.update(B, dB);
                    
                    this.Weights_tensor.Add(W, layer);
                    this.Bias_tensor.Add(B, layer);
                  }
             }
         }
        else //Batch Gradient Descent
          {
               
            for (uint batch=0, batch_start=0, batch_end=batch_size; batch<num_batches; batch++, batch_start+=batch_size, batch_end=(batch_start+batch_size-1))
               {
                  matrix batch_x = MatrixExtend::Get(x, batch_start, batch_end-1);
                  vector batch_y = MatrixExtend::Get(y, batch_start, batch_end-1);
                  
                  rows = batch_x.Rows();              
                  
                    for (ulong iter=0; iter<rows ; iter++) //iterate through all data points
                      {
                        pred_v[0] = predict(batch_x.Row(iter));
                        actual_v[0] = y[iter];
                        
            //---
                        
                      DELTA.Resize(mlp.outputs,1);
                      
                      for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                        {    
                           Partial_Derivatives = this.Output_tensor.Get(int(layer));
                           temp_inputs = this.Input_tensor.Get(int(layer));
                           
                           Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                           
                           if (mlp.hidden_layers-1 == layer) //Last layer
                            {                     
                              LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                              
                              DELTA.Col(LossGradient, 0);
                            }
                            
                           else
                            {
                              W = this.Weights_tensor.Get(layer+1);
                              
                              DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                            }
                          
                          //-- Observation | DeLTA matrix is same size as the bias matrix
                          
                          W = this.Weights_tensor.Get(layer);
                          B = this.Bias_tensor.Get(layer);
                        
                         //--- Derivatives wrt weights and bias
                        
                          dB = DELTA;
                          dW = DELTA.MatMul(temp_inputs.Transpose());                   
                          
                         //--- Weights updates
                          
                          optimizer_weights.update(W, dW);
                          optimizer_bias.update(B, dB);
                          
                          this.Weights_tensor.Add(W, layer);
                          this.Bias_tensor.Add(B, layer);
                        }
                    }
                 
                 pred_v = predict(batch_x);
                 
                 batch_loss[batch] = pred_v.Loss(batch_y, ENUM_LOSS_FUNCTION(m_loss_function));
                 batch_loss[batch] = MathIsValidNumber(batch_loss[batch]) ? (batch_loss[batch]>1e6 ? 1e6 : batch_loss[batch]) : 1e6; //Check for nan and return some large value if it is nan
                 
                 batch_accuracy[batch] = Metrics::r_squared(batch_y, pred_v);
                 
                 if (show_batch_progress)
                  printf("----> batch[%d/%d] batch-loss %.5f accuracy %.3f",batch+1,num_batches,batch_loss[batch], batch_accuracy[batch]);  
              }
          }
          
//--- End of an epoch
      
        vector validation_loss(cv_tensor.SIZE);
        vector validation_acc(cv_tensor.SIZE);
        for (ulong i=0; i<cv_tensor.SIZE; i++)
          {
            validation_data = cv_tensor.Get(i);
            MatrixExtend::XandYSplitMatrices(validation_data, validation_x, validation_y);
            
            vector val_preds = this.predict(validation_x);;
            
            validation_loss[i] = val_preds.Loss(validation_y, ENUM_LOSS_FUNCTION(m_loss_function));
            validation_acc[i] = Metrics::r_squared(validation_y, val_preds);
          }
                  
        pred_v = this.predict(x);
        
        if (batch_size==0)
          {      
              backprop_struct.training_loss[epoch] = pred_v.Loss(y, ENUM_LOSS_FUNCTION(m_loss_function));
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
        else
          {
              backprop_struct.training_loss[epoch] = batch_loss.Mean();
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
          
        double epoch_stop = GetTickCount();  
        printf("--> Epoch [%d/%d] training -> loss %.8f accuracy %.3f validation -> loss %.5f accuracy %.3f | Elapsed %s ",epoch+1,epochs,backprop_struct.training_loss[epoch],Metrics::r_squared(y, pred_v),backprop_struct.validation_loss[epoch],validation_acc.Mean(),this.ConvertTime((epoch_stop-epoch_start)/1000.0));
     }
     
   isBackProp = false;
  
  
  if (CheckPointer(this.Input_tensor) != POINTER_INVALID)  delete(this.Input_tensor);
  if (CheckPointer(this.Output_tensor) != POINTER_INVALID)  delete(this.Output_tensor); 
  if (CheckPointer(optimizer)!=POINTER_INVALID)  
    delete optimizer;
    
   return backprop_struct;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
backprop CRegressorNets::backpropagation(const matrix& x, const vector &y, OptimizerAdaDelta *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
   isBackProp = true;
   
//---

   backprop backprop_struct;
   backprop_struct.Init(epochs);
   
   ulong rows = x.Rows();
   
   mlp.inputs = x.Cols();
   mlp.outputs = 1;
   
//---

   vector v2 = {(double)mlp.outputs}; //Adding the output layer to the mix of hidden layers
  
   HL_CONFIG = MatrixExtend::concatenate(HL_CONFIG, v2);
   mlp.hidden_layers = HL_CONFIG.Size();
   W_CONFIG.Resize(HL_CONFIG.Size());
     
//---

   if (y.Size() != rows)
     {
        Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
        return backprop_struct;
     }
     
     
     matrix W, B;
     
//--- GENERATE WEIGHTS
    
     this.Weights_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Bias_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Input_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Output_tensor = new CTensors((uint)mlp.hidden_layers);
     
     ulong layer_input = mlp.inputs; 
     
     for (ulong i=0; i<mlp.hidden_layers; i++)
       {
          W_CONFIG[i] = layer_input*HL_CONFIG[i];
          
          W = MatrixExtend::Random(0.0, 1.0,(ulong)HL_CONFIG[i],layer_input, m_random_state);
          
          W = W * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Weights_tensor.Add(W, i);
          
          B = MatrixExtend::Random(0.0, 0.5,(ulong)HL_CONFIG[i],1,m_random_state);
          
          B = B * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Bias_tensor.Add(B, i);
          
          layer_input = (ulong)HL_CONFIG[i];
       }
     
//---

   if (MQLInfoInteger(MQL_DEBUG))
      Comment("<-------------------  R E G R E S S O R   N E T S  ------------------------->\n",
            "HL_CONFIG ",HL_CONFIG," TOTAL HL(S) ",mlp.hidden_layers,"\n",
            "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
            "NN INPUTS ",mlp.inputs," OUTPUT ",mlp.outputs
           );

//--- Optimizer
      
   OptimizerAdaDelta optimizer_weights = optimizer;
   OptimizerAdaDelta optimizer_bias = optimizer;
   
   if (batch_size>0)
    {
      OptimizerMinBGD optimizer_weights;
      OptimizerMinBGD optimizer_bias;
    }
     
//--- Cross validation

    CCrossValidation cross_validation;      
    CTensors *cv_tensor;
    matrix validation_data = MatrixExtend::concatenate(x, y);
    matrix validation_x;
    vector validation_y;
    
    cv_tensor = cross_validation.KFoldCV(validation_data, 10); //k-fold cross validation | 10 folds selected
    
//---

    matrix DELTA = {};
    double actual=0, pred=0;
    
    matrix temp_inputs ={};
    
    matrix dB = {}; //Bias Derivatives
    matrix dW = {}; //Weight Derivatives
    
   
    for (ulong epoch=0; epoch<epochs && !IsStopped(); epoch++)
      {        
        double epoch_start = GetTickCount(); 

        uint num_batches = (uint)MathFloor(x.Rows()/(batch_size+DBL_EPSILON));
        
        vector batch_loss(num_batches), 
               batch_accuracy(num_batches);
                       
         vector actual_v(1), pred_v(1), LossGradient = {};
         
         if (batch_size==0) //Stochastic Gradient Descent
          {
           for (ulong iter=0; iter<rows; iter++) //iterate through all data points
             {
               pred = predict(x.Row(iter));
               actual = y[iter];
               
               pred_v[0] = pred; 
               actual_v[0] = actual; 
   //---
                
                DELTA.Resize(mlp.outputs,1);
                
                for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                  {    
                     Partial_Derivatives = this.Output_tensor.Get(int(layer));
                     temp_inputs = this.Input_tensor.Get(int(layer));
                     
                     Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                     
                     if (mlp.hidden_layers-1 == layer) //Last layer
                      {                     
                        LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                        
                        DELTA.Col(LossGradient, 0);
                      }
                      
                     else
                      {
                        W = this.Weights_tensor.Get(layer+1);
                        
                        DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                      }
                    
                    //-- Observation | DeLTA matrix is same size as the bias matrix
                    
                    W = this.Weights_tensor.Get(layer);
                    B = this.Bias_tensor.Get(layer);
                  
                   //--- Derivatives wrt weights and bias
                  
                    dB = DELTA;
                    dW = DELTA.MatMul(temp_inputs.Transpose());                   
                    
                   //--- Weights updates
                    
                    optimizer_weights.update(W, dW);
                    optimizer_bias.update(B, dB);
                    
                    this.Weights_tensor.Add(W, layer);
                    this.Bias_tensor.Add(B, layer);
                  }
             }
         }
        else //Batch Gradient Descent
          {
               
            for (uint batch=0, batch_start=0, batch_end=batch_size; batch<num_batches; batch++, batch_start+=batch_size, batch_end=(batch_start+batch_size-1))
               {
                  matrix batch_x = MatrixExtend::Get(x, batch_start, batch_end-1);
                  vector batch_y = MatrixExtend::Get(y, batch_start, batch_end-1);
                  
                  rows = batch_x.Rows();              
                  
                    for (ulong iter=0; iter<rows ; iter++) //iterate through all data points
                      {
                        pred_v[0] = predict(batch_x.Row(iter));
                        actual_v[0] = y[iter];
                        
            //---
                        
                      DELTA.Resize(mlp.outputs,1);
                      
                      for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                        {    
                           Partial_Derivatives = this.Output_tensor.Get(int(layer));
                           temp_inputs = this.Input_tensor.Get(int(layer));
                           
                           Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                           
                           if (mlp.hidden_layers-1 == layer) //Last layer
                            {                     
                              LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                              
                              DELTA.Col(LossGradient, 0);
                            }
                            
                           else
                            {
                              W = this.Weights_tensor.Get(layer+1);
                              
                              DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                            }
                          
                          //-- Observation | DeLTA matrix is same size as the bias matrix
                          
                          W = this.Weights_tensor.Get(layer);
                          B = this.Bias_tensor.Get(layer);
                        
                         //--- Derivatives wrt weights and bias
                        
                          dB = DELTA;
                          dW = DELTA.MatMul(temp_inputs.Transpose());                   
                          
                         //--- Weights updates
                          
                          optimizer_weights.update(W, dW);
                          optimizer_bias.update(B, dB);
                          
                          this.Weights_tensor.Add(W, layer);
                          this.Bias_tensor.Add(B, layer);
                        }
                    }
                 
                 pred_v = predict(batch_x);
                 
                 batch_loss[batch] = pred_v.Loss(batch_y, ENUM_LOSS_FUNCTION(m_loss_function));
                 batch_loss[batch] = MathIsValidNumber(batch_loss[batch]) ? (batch_loss[batch]>1e6 ? 1e6 : batch_loss[batch]) : 1e6; //Check for nan and return some large value if it is nan
                 
                 batch_accuracy[batch] = Metrics::r_squared(batch_y, pred_v);
                 
                 if (show_batch_progress)
                  printf("----> batch[%d/%d] batch-loss %.5f accuracy %.3f",batch+1,num_batches,batch_loss[batch], batch_accuracy[batch]);  
              }
          }
          
//--- End of an epoch
      
        vector validation_loss(cv_tensor.SIZE);
        vector validation_acc(cv_tensor.SIZE);
        for (ulong i=0; i<cv_tensor.SIZE; i++)
          {
            validation_data = cv_tensor.Get(i);
            MatrixExtend::XandYSplitMatrices(validation_data, validation_x, validation_y);
            
            vector val_preds = this.predict(validation_x);;
            
            validation_loss[i] = val_preds.Loss(validation_y, ENUM_LOSS_FUNCTION(m_loss_function));
            validation_acc[i] = Metrics::r_squared(validation_y, val_preds);
          }
                  
        pred_v = this.predict(x);
        
        if (batch_size==0)
          {      
              backprop_struct.training_loss[epoch] = pred_v.Loss(y, ENUM_LOSS_FUNCTION(m_loss_function));
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
        else
          {
              backprop_struct.training_loss[epoch] = batch_loss.Mean();
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
          
        double epoch_stop = GetTickCount();  
        printf("--> Epoch [%d/%d] training -> loss %.8f accuracy %.3f validation -> loss %.5f accuracy %.3f | Elapsed %s ",epoch+1,epochs,backprop_struct.training_loss[epoch],Metrics::r_squared(y, pred_v),backprop_struct.validation_loss[epoch],validation_acc.Mean(),this.ConvertTime((epoch_stop-epoch_start)/1000.0));
     }
     
   isBackProp = false;
   
  if (CheckPointer(this.Input_tensor) != POINTER_INVALID)  delete(this.Input_tensor);
  if (CheckPointer(this.Output_tensor) != POINTER_INVALID)  delete(this.Output_tensor); 
  if (CheckPointer(optimizer)!=POINTER_INVALID)  
    delete optimizer;
    
   return backprop_struct;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
backprop CRegressorNets::backpropagation(const matrix& x, const vector &y, OptimizerAdaGrad *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
   isBackProp = true;
   
//---

   backprop backprop_struct;
   backprop_struct.Init(epochs);
   
   ulong rows = x.Rows();
   
   mlp.inputs = x.Cols();
   mlp.outputs = 1;
   
//---

   vector v2 = {(double)mlp.outputs}; //Adding the output layer to the mix of hidden layers
  
   HL_CONFIG = MatrixExtend::concatenate(HL_CONFIG, v2);
   mlp.hidden_layers = HL_CONFIG.Size();
   W_CONFIG.Resize(HL_CONFIG.Size());
     
//---

   if (y.Size() != rows)
     {
        Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
        return backprop_struct;
     }
     
     
     matrix W, B;
     
//--- GENERATE WEIGHTS
    
     this.Weights_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Bias_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Input_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Output_tensor = new CTensors((uint)mlp.hidden_layers);
     
     ulong layer_input = mlp.inputs; 
     
     for (ulong i=0; i<mlp.hidden_layers; i++)
       {
          W_CONFIG[i] = layer_input*HL_CONFIG[i];
          
          W = MatrixExtend::Random(0.0, 1.0,(ulong)HL_CONFIG[i],layer_input, m_random_state);
          
          W = W * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Weights_tensor.Add(W, i);
          
          B = MatrixExtend::Random(0.0, 0.5,(ulong)HL_CONFIG[i],1,m_random_state);
          
          B = B * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Bias_tensor.Add(B, i);
          
          layer_input = (ulong)HL_CONFIG[i];
       }
     
//---

   if (MQLInfoInteger(MQL_DEBUG))
      Comment("<-------------------  R E G R E S S O R   N E T S  ------------------------->\n",
            "HL_CONFIG ",HL_CONFIG," TOTAL HL(S) ",mlp.hidden_layers,"\n",
            "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
            "NN INPUTS ",mlp.inputs," OUTPUT ",mlp.outputs
           );

//--- Optimizer
      
   OptimizerAdaGrad optimizer_weights = optimizer;
   OptimizerAdaGrad optimizer_bias = optimizer;
   
   if (batch_size>0)
    {
      OptimizerMinBGD optimizer_weights;
      OptimizerMinBGD optimizer_bias;
    }
     
//--- Cross validation

    CCrossValidation cross_validation;      
    CTensors *cv_tensor;
    matrix validation_data = MatrixExtend::concatenate(x, y);
    matrix validation_x;
    vector validation_y;
    
    cv_tensor = cross_validation.KFoldCV(validation_data, 10); //k-fold cross validation | 10 folds selected
    
//---

    matrix DELTA = {};
    double actual=0, pred=0;
    
    matrix temp_inputs ={};
    
    matrix dB = {}; //Bias Derivatives
    matrix dW = {}; //Weight Derivatives
    
   
    for (ulong epoch=0; epoch<epochs && !IsStopped(); epoch++)
      {        
        double epoch_start = GetTickCount(); 

        uint num_batches = (uint)MathFloor(x.Rows()/(batch_size+DBL_EPSILON));
        
        vector batch_loss(num_batches), 
               batch_accuracy(num_batches);
                       
         vector actual_v(1), pred_v(1), LossGradient = {};
         
         if (batch_size==0) //Stochastic Gradient Descent
          {
           for (ulong iter=0; iter<rows; iter++) //iterate through all data points
             {
               pred = predict(x.Row(iter));
               actual = y[iter];
               
               pred_v[0] = pred; 
               actual_v[0] = actual; 
   //---
                
                DELTA.Resize(mlp.outputs,1);
                
                for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                  {    
                     Partial_Derivatives = this.Output_tensor.Get(int(layer));
                     temp_inputs = this.Input_tensor.Get(int(layer));
                     
                     Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                     
                     if (mlp.hidden_layers-1 == layer) //Last layer
                      {                     
                        LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                        
                        DELTA.Col(LossGradient, 0);
                      }
                      
                     else
                      {
                        W = this.Weights_tensor.Get(layer+1);
                        
                        DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                      }
                    
                    //-- Observation | DeLTA matrix is same size as the bias matrix
                    
                    W = this.Weights_tensor.Get(layer);
                    B = this.Bias_tensor.Get(layer);
                  
                   //--- Derivatives wrt weights and bias
                  
                    dB = DELTA;
                    dW = DELTA.MatMul(temp_inputs.Transpose());                   
                    
                   //--- Weights updates
                    
                    optimizer_weights.update(W, dW);
                    optimizer_bias.update(B, dB);
                    
                    this.Weights_tensor.Add(W, layer);
                    this.Bias_tensor.Add(B, layer);
                  }
             }
         }
        else //Batch Gradient Descent
          {
               
            for (uint batch=0, batch_start=0, batch_end=batch_size; batch<num_batches; batch++, batch_start+=batch_size, batch_end=(batch_start+batch_size-1))
               {
                  matrix batch_x = MatrixExtend::Get(x, batch_start, batch_end-1);
                  vector batch_y = MatrixExtend::Get(y, batch_start, batch_end-1);
                  
                  rows = batch_x.Rows();              
                  
                    for (ulong iter=0; iter<rows ; iter++) //iterate through all data points
                      {
                        pred_v[0] = predict(batch_x.Row(iter));
                        actual_v[0] = y[iter];
                        
            //---
                        
                      DELTA.Resize(mlp.outputs,1);
                      
                      for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                        {    
                           Partial_Derivatives = this.Output_tensor.Get(int(layer));
                           temp_inputs = this.Input_tensor.Get(int(layer));
                           
                           Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                           
                           if (mlp.hidden_layers-1 == layer) //Last layer
                            {                     
                              LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                              
                              DELTA.Col(LossGradient, 0);
                            }
                            
                           else
                            {
                              W = this.Weights_tensor.Get(layer+1);
                              
                              DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                            }
                          
                          //-- Observation | DeLTA matrix is same size as the bias matrix
                          
                          W = this.Weights_tensor.Get(layer);
                          B = this.Bias_tensor.Get(layer);
                        
                         //--- Derivatives wrt weights and bias
                        
                          dB = DELTA;
                          dW = DELTA.MatMul(temp_inputs.Transpose());                   
                          
                         //--- Weights updates
                          
                          optimizer_weights.update(W, dW);
                          optimizer_bias.update(B, dB);
                          
                          this.Weights_tensor.Add(W, layer);
                          this.Bias_tensor.Add(B, layer);
                        }
                    }
                 
                 pred_v = predict(batch_x);
                 
                 batch_loss[batch] = pred_v.Loss(batch_y, ENUM_LOSS_FUNCTION(m_loss_function));
                 batch_loss[batch] = MathIsValidNumber(batch_loss[batch]) ? (batch_loss[batch]>1e6 ? 1e6 : batch_loss[batch]) : 1e6; //Check for nan and return some large value if it is nan
                 
                 batch_accuracy[batch] = Metrics::r_squared(batch_y, pred_v);
                 
                 if (show_batch_progress)
                  printf("----> batch[%d/%d] batch-loss %.5f accuracy %.3f",batch+1,num_batches,batch_loss[batch], batch_accuracy[batch]);  
              }
          }
          
//--- End of an epoch
      
        vector validation_loss(cv_tensor.SIZE);
        vector validation_acc(cv_tensor.SIZE);
        for (ulong i=0; i<cv_tensor.SIZE; i++)
          {
            validation_data = cv_tensor.Get(i);
            MatrixExtend::XandYSplitMatrices(validation_data, validation_x, validation_y);
            
            vector val_preds = this.predict(validation_x);;
            
            validation_loss[i] = val_preds.Loss(validation_y, ENUM_LOSS_FUNCTION(m_loss_function));
            validation_acc[i] = Metrics::r_squared(validation_y, val_preds);
          }
                  
        pred_v = this.predict(x);
        
        if (batch_size==0)
          {      
              backprop_struct.training_loss[epoch] = pred_v.Loss(y, ENUM_LOSS_FUNCTION(m_loss_function));
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
        else
          {
              backprop_struct.training_loss[epoch] = batch_loss.Mean();
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
          
        double epoch_stop = GetTickCount();  
        printf("--> Epoch [%d/%d] training -> loss %.8f accuracy %.3f validation -> loss %.5f accuracy %.3f | Elapsed %s ",epoch+1,epochs,backprop_struct.training_loss[epoch],Metrics::r_squared(y, pred_v),backprop_struct.validation_loss[epoch],validation_acc.Mean(),this.ConvertTime((epoch_stop-epoch_start)/1000.0));
     }
     
   isBackProp = false;
   
  if (CheckPointer(this.Input_tensor) != POINTER_INVALID)  delete(this.Input_tensor);
  if (CheckPointer(this.Output_tensor) != POINTER_INVALID)  delete(this.Output_tensor); 
  if (CheckPointer(optimizer)!=POINTER_INVALID)  
    delete optimizer;
    
   return backprop_struct;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
backprop CRegressorNets::backpropagation(const matrix& x, const vector &y, OptimizerAdam *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
   isBackProp = true;
   
//---

   backprop backprop_struct;
   backprop_struct.Init(epochs);
   
   ulong rows = x.Rows();
   
   mlp.inputs = x.Cols();
   mlp.outputs = 1;
   
//---

   vector v2 = {(double)mlp.outputs}; //Adding the output layer to the mix of hidden layers
  
   HL_CONFIG = MatrixExtend::concatenate(HL_CONFIG, v2);
   mlp.hidden_layers = HL_CONFIG.Size();
   W_CONFIG.Resize(HL_CONFIG.Size());
     
//---

   if (y.Size() != rows)
     {
        Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
        return backprop_struct;
     }
     
     
     matrix W, B;
     
//--- GENERATE WEIGHTS
    
     this.Weights_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Bias_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Input_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Output_tensor = new CTensors((uint)mlp.hidden_layers);
     
     ulong layer_input = mlp.inputs; 
     
     for (ulong i=0; i<mlp.hidden_layers; i++)
       {
          W_CONFIG[i] = layer_input*HL_CONFIG[i];
          
          W = MatrixExtend::Random(0.0, 1.0,(ulong)HL_CONFIG[i],layer_input, m_random_state);
          
          W = W * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Weights_tensor.Add(W, i);
          
          B = MatrixExtend::Random(0.0, 0.5,(ulong)HL_CONFIG[i],1,m_random_state);
          
          B = B * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Bias_tensor.Add(B, i);
          
          layer_input = (ulong)HL_CONFIG[i];
       }
     
//---

   if (MQLInfoInteger(MQL_DEBUG))
      Comment("<-------------------  R E G R E S S O R   N E T S  ------------------------->\n",
            "HL_CONFIG ",HL_CONFIG," TOTAL HL(S) ",mlp.hidden_layers,"\n",
            "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
            "NN INPUTS ",mlp.inputs," OUTPUT ",mlp.outputs
           );

//--- Optimizer
      
   OptimizerAdam optimizer_weights = optimizer;
   OptimizerAdam optimizer_bias = optimizer;
   
   if (batch_size>0)
    {
      OptimizerMinBGD optimizer_weights;
      OptimizerMinBGD optimizer_bias;
    }
     
//--- Cross validation

    CCrossValidation cross_validation;      
    CTensors *cv_tensor;
    matrix validation_data = MatrixExtend::concatenate(x, y);
    matrix validation_x;
    vector validation_y;
    
    cv_tensor = cross_validation.KFoldCV(validation_data, 10); //k-fold cross validation | 10 folds selected
    
//---

    matrix DELTA = {};
    double actual=0, pred=0;
    
    matrix temp_inputs ={};
    
    matrix dB = {}; //Bias Derivatives
    matrix dW = {}; //Weight Derivatives
    
   
    for (ulong epoch=0; epoch<epochs && !IsStopped(); epoch++)
      {        
        double epoch_start = GetTickCount(); 

        uint num_batches = (uint)MathFloor(x.Rows()/(batch_size+DBL_EPSILON));
        
        vector batch_loss(num_batches), 
               batch_accuracy(num_batches);
                       
         vector actual_v(1), pred_v(1), LossGradient = {};
         
         if (batch_size==0) //Stochastic Gradient Descent
          {
           for (ulong iter=0; iter<rows; iter++) //iterate through all data points
             {
               pred = predict(x.Row(iter));
               actual = y[iter];
               
               pred_v[0] = pred; 
               actual_v[0] = actual; 
   //---
                
                DELTA.Resize(mlp.outputs,1);
                
                for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                  {    
                     Partial_Derivatives = this.Output_tensor.Get(int(layer));
                     temp_inputs = this.Input_tensor.Get(int(layer));
                     
                     Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                     
                     if (mlp.hidden_layers-1 == layer) //Last layer
                      {                     
                        LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                        
                        DELTA.Col(LossGradient, 0);
                      }
                      
                     else
                      {
                        W = this.Weights_tensor.Get(layer+1);
                        
                        DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                      }
                    
                    //-- Observation | DeLTA matrix is same size as the bias matrix
                    
                    W = this.Weights_tensor.Get(layer);
                    B = this.Bias_tensor.Get(layer);
                  
                   //--- Derivatives wrt weights and bias
                  
                    dB = DELTA;
                    dW = DELTA.MatMul(temp_inputs.Transpose());                   
                    
                   //--- Weights updates
                    
                    optimizer_weights.update(W, dW);
                    optimizer_bias.update(B, dB);
                    
                    this.Weights_tensor.Add(W, layer);
                    this.Bias_tensor.Add(B, layer);
                  }
             }
         }
        else //Batch Gradient Descent
          {
               
            for (uint batch=0, batch_start=0, batch_end=batch_size; batch<num_batches; batch++, batch_start+=batch_size, batch_end=(batch_start+batch_size-1))
               {
                  matrix batch_x = MatrixExtend::Get(x, batch_start, batch_end-1);
                  vector batch_y = MatrixExtend::Get(y, batch_start, batch_end-1);
                  
                  rows = batch_x.Rows();              
                  
                    for (ulong iter=0; iter<rows ; iter++) //iterate through all data points
                      {
                        pred_v[0] = predict(batch_x.Row(iter));
                        actual_v[0] = y[iter];
                        
            //---
                        
                      DELTA.Resize(mlp.outputs,1);
                      
                      for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                        {    
                           Partial_Derivatives = this.Output_tensor.Get(int(layer));
                           temp_inputs = this.Input_tensor.Get(int(layer));
                           
                           Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                           
                           if (mlp.hidden_layers-1 == layer) //Last layer
                            {                     
                              LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                              
                              DELTA.Col(LossGradient, 0);
                            }
                            
                           else
                            {
                              W = this.Weights_tensor.Get(layer+1);
                              
                              DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                            }
                          
                          //-- Observation | DeLTA matrix is same size as the bias matrix
                          
                          W = this.Weights_tensor.Get(layer);
                          B = this.Bias_tensor.Get(layer);
                        
                         //--- Derivatives wrt weights and bias
                        
                          dB = DELTA;
                          dW = DELTA.MatMul(temp_inputs.Transpose());                   
                          
                         //--- Weights updates
                          
                          optimizer_weights.update(W, dW);
                          optimizer_bias.update(B, dB);
                          
                          this.Weights_tensor.Add(W, layer);
                          this.Bias_tensor.Add(B, layer);
                        }
                    }
                 
                 pred_v = predict(batch_x);
                 
                 batch_loss[batch] = pred_v.Loss(batch_y, ENUM_LOSS_FUNCTION(m_loss_function));
                 batch_loss[batch] = MathIsValidNumber(batch_loss[batch]) ? (batch_loss[batch]>1e6 ? 1e6 : batch_loss[batch]) : 1e6; //Check for nan and return some large value if it is nan
                 
                 batch_accuracy[batch] = Metrics::r_squared(batch_y, pred_v);
                 
                 if (show_batch_progress)
                  printf("----> batch[%d/%d] batch-loss %.5f accuracy %.3f",batch+1,num_batches,batch_loss[batch], batch_accuracy[batch]);  
              }
          }
          
//--- End of an epoch
      
        vector validation_loss(cv_tensor.SIZE);
        vector validation_acc(cv_tensor.SIZE);
        for (ulong i=0; i<cv_tensor.SIZE; i++)
          {
            validation_data = cv_tensor.Get(i);
            MatrixExtend::XandYSplitMatrices(validation_data, validation_x, validation_y);
            
            vector val_preds = this.predict(validation_x);;
            
            validation_loss[i] = val_preds.Loss(validation_y, ENUM_LOSS_FUNCTION(m_loss_function));
            validation_acc[i] = Metrics::r_squared(validation_y, val_preds);
          }
                  
        pred_v = this.predict(x);
        
        if (batch_size==0)
          {      
              backprop_struct.training_loss[epoch] = pred_v.Loss(y, ENUM_LOSS_FUNCTION(m_loss_function));
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
        else
          {
              backprop_struct.training_loss[epoch] = batch_loss.Mean();
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
          
        double epoch_stop = GetTickCount();  
        printf("--> Epoch [%d/%d] training -> loss %.8f accuracy %.3f validation -> loss %.5f accuracy %.3f | Elapsed %s ",epoch+1,epochs,backprop_struct.training_loss[epoch],Metrics::r_squared(y, pred_v),backprop_struct.validation_loss[epoch],validation_acc.Mean(),this.ConvertTime((epoch_stop-epoch_start)/1000.0));
     }
     
   isBackProp = false;
   
  if (CheckPointer(this.Input_tensor) != POINTER_INVALID)  delete(this.Input_tensor);
  if (CheckPointer(this.Output_tensor) != POINTER_INVALID)  delete(this.Output_tensor); 
  if (CheckPointer(optimizer)!=POINTER_INVALID)  
    delete optimizer;
    
   return backprop_struct;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
backprop CRegressorNets::backpropagation(const matrix& x, const vector &y, OptimizerNadam *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
   isBackProp = true;
   
//---

   backprop backprop_struct;
   backprop_struct.Init(epochs);
   
   ulong rows = x.Rows();
   
   mlp.inputs = x.Cols();
   mlp.outputs = 1;
   
//---

   vector v2 = {(double)mlp.outputs}; //Adding the output layer to the mix of hidden layers
  
   HL_CONFIG = MatrixExtend::concatenate(HL_CONFIG, v2);
   mlp.hidden_layers = HL_CONFIG.Size();
   W_CONFIG.Resize(HL_CONFIG.Size());
     
//---

   if (y.Size() != rows)
     {
        Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
        return backprop_struct;
     }
     
     
     matrix W, B;
     
//--- GENERATE WEIGHTS
    
     this.Weights_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Bias_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Input_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Output_tensor = new CTensors((uint)mlp.hidden_layers);
     
     ulong layer_input = mlp.inputs; 
     
     for (ulong i=0; i<mlp.hidden_layers; i++)
       {
          W_CONFIG[i] = layer_input*HL_CONFIG[i];
          
          W = MatrixExtend::Random(0.0, 1.0,(ulong)HL_CONFIG[i],layer_input, m_random_state);
          
          W = W * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Weights_tensor.Add(W, i);
          
          B = MatrixExtend::Random(0.0, 0.5,(ulong)HL_CONFIG[i],1,m_random_state);
          
          B = B * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Bias_tensor.Add(B, i);
          
          layer_input = (ulong)HL_CONFIG[i];
       }
     
//---

   if (MQLInfoInteger(MQL_DEBUG))
      Comment("<-------------------  R E G R E S S O R   N E T S  ------------------------->\n",
            "HL_CONFIG ",HL_CONFIG," TOTAL HL(S) ",mlp.hidden_layers,"\n",
            "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
            "NN INPUTS ",mlp.inputs," OUTPUT ",mlp.outputs
           );

//--- Optimizer
      
   OptimizerNadam optimizer_weights = optimizer;
   OptimizerNadam optimizer_bias = optimizer;
   
   if (batch_size>0)
    {
      OptimizerMinBGD optimizer_weights;
      OptimizerMinBGD optimizer_bias;
    }
     
//--- Cross validation

    CCrossValidation cross_validation;      
    CTensors *cv_tensor;
    matrix validation_data = MatrixExtend::concatenate(x, y);
    matrix validation_x;
    vector validation_y;
    
    cv_tensor = cross_validation.KFoldCV(validation_data, 10); //k-fold cross validation | 10 folds selected
    
//---

    matrix DELTA = {};
    double actual=0, pred=0;
    
    matrix temp_inputs ={};
    
    matrix dB = {}; //Bias Derivatives
    matrix dW = {}; //Weight Derivatives
    
   
    for (ulong epoch=0; epoch<epochs && !IsStopped(); epoch++)
      {        
        double epoch_start = GetTickCount(); 

        uint num_batches = (uint)MathFloor(x.Rows()/(batch_size+DBL_EPSILON));
        
        vector batch_loss(num_batches), 
               batch_accuracy(num_batches);
                       
         vector actual_v(1), pred_v(1), LossGradient = {};
         
         if (batch_size==0) //Stochastic Gradient Descent
          {
           for (ulong iter=0; iter<rows; iter++) //iterate through all data points
             {
               pred = predict(x.Row(iter));
               actual = y[iter];
               
               pred_v[0] = pred; 
               actual_v[0] = actual; 
   //---
                
                DELTA.Resize(mlp.outputs,1);
                
                for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                  {    
                     Partial_Derivatives = this.Output_tensor.Get(int(layer));
                     temp_inputs = this.Input_tensor.Get(int(layer));
                     
                     Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                     
                     if (mlp.hidden_layers-1 == layer) //Last layer
                      {                     
                        LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                        
                        DELTA.Col(LossGradient, 0);
                      }
                      
                     else
                      {
                        W = this.Weights_tensor.Get(layer+1);
                        
                        DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                      }
                    
                    //-- Observation | DeLTA matrix is same size as the bias matrix
                    
                    W = this.Weights_tensor.Get(layer);
                    B = this.Bias_tensor.Get(layer);
                  
                   //--- Derivatives wrt weights and bias
                  
                    dB = DELTA;
                    dW = DELTA.MatMul(temp_inputs.Transpose());                   
                    
                   //--- Weights updates
                    
                    optimizer_weights.update(W, dW);
                    optimizer_bias.update(B, dB);
                    
                    this.Weights_tensor.Add(W, layer);
                    this.Bias_tensor.Add(B, layer);
                  }
             }
         }
        else //Batch Gradient Descent
          {
               
            for (uint batch=0, batch_start=0, batch_end=batch_size; batch<num_batches; batch++, batch_start+=batch_size, batch_end=(batch_start+batch_size-1))
               {
                  matrix batch_x = MatrixExtend::Get(x, batch_start, batch_end-1);
                  vector batch_y = MatrixExtend::Get(y, batch_start, batch_end-1);
                  
                  rows = batch_x.Rows();              
                  
                    for (ulong iter=0; iter<rows ; iter++) //iterate through all data points
                      {
                        pred_v[0] = predict(batch_x.Row(iter));
                        actual_v[0] = y[iter];
                        
            //---
                        
                      DELTA.Resize(mlp.outputs,1);
                      
                      for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                        {    
                           Partial_Derivatives = this.Output_tensor.Get(int(layer));
                           temp_inputs = this.Input_tensor.Get(int(layer));
                           
                           Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                           
                           if (mlp.hidden_layers-1 == layer) //Last layer
                            {                     
                              LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                              
                              DELTA.Col(LossGradient, 0);
                            }
                            
                           else
                            {
                              W = this.Weights_tensor.Get(layer+1);
                              
                              DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                            }
                          
                          //-- Observation | DeLTA matrix is same size as the bias matrix
                          
                          W = this.Weights_tensor.Get(layer);
                          B = this.Bias_tensor.Get(layer);
                        
                         //--- Derivatives wrt weights and bias
                        
                          dB = DELTA;
                          dW = DELTA.MatMul(temp_inputs.Transpose());                   
                          
                         //--- Weights updates
                          
                          optimizer_weights.update(W, dW);
                          optimizer_bias.update(B, dB);
                          
                          this.Weights_tensor.Add(W, layer);
                          this.Bias_tensor.Add(B, layer);
                        }
                    }
                 
                 pred_v = predict(batch_x);
                 
                 batch_loss[batch] = pred_v.Loss(batch_y, ENUM_LOSS_FUNCTION(m_loss_function));
                 batch_loss[batch] = MathIsValidNumber(batch_loss[batch]) ? (batch_loss[batch]>1e6 ? 1e6 : batch_loss[batch]) : 1e6; //Check for nan and return some large value if it is nan
                 
                 batch_accuracy[batch] = Metrics::r_squared(batch_y, pred_v);
                 
                 if (show_batch_progress)
                  printf("----> batch[%d/%d] batch-loss %.5f accuracy %.3f",batch+1,num_batches,batch_loss[batch], batch_accuracy[batch]);  
              }
          }
          
//--- End of an epoch
      
        vector validation_loss(cv_tensor.SIZE);
        vector validation_acc(cv_tensor.SIZE);
        for (ulong i=0; i<cv_tensor.SIZE; i++)
          {
            validation_data = cv_tensor.Get(i);
            MatrixExtend::XandYSplitMatrices(validation_data, validation_x, validation_y);
            
            vector val_preds = this.predict(validation_x);;
            
            validation_loss[i] = val_preds.Loss(validation_y, ENUM_LOSS_FUNCTION(m_loss_function));
            validation_acc[i] = Metrics::r_squared(validation_y, val_preds);
          }
                  
        pred_v = this.predict(x);
        
        if (batch_size==0)
          {      
              backprop_struct.training_loss[epoch] = pred_v.Loss(y, ENUM_LOSS_FUNCTION(m_loss_function));
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
        else
          {
              backprop_struct.training_loss[epoch] = batch_loss.Mean();
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
          
        double epoch_stop = GetTickCount();  
        printf("--> Epoch [%d/%d] training -> loss %.8f accuracy %.3f validation -> loss %.5f accuracy %.3f | Elapsed %s ",epoch+1,epochs,backprop_struct.training_loss[epoch],Metrics::r_squared(y, pred_v),backprop_struct.validation_loss[epoch],validation_acc.Mean(),this.ConvertTime((epoch_stop-epoch_start)/1000.0));
     }
     
   isBackProp = false;
   
  if (CheckPointer(this.Input_tensor) != POINTER_INVALID)  delete(this.Input_tensor);
  if (CheckPointer(this.Output_tensor) != POINTER_INVALID)  delete(this.Output_tensor); 
  if (CheckPointer(optimizer)!=POINTER_INVALID)  
    delete optimizer;
    
   return backprop_struct;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
backprop CRegressorNets::backpropagation(const matrix& x, const vector &y, OptimizerRMSprop *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
   isBackProp = true;
   
//---

   backprop backprop_struct;
   backprop_struct.Init(epochs);
   
   ulong rows = x.Rows();
   
   mlp.inputs = x.Cols();
   mlp.outputs = 1;
   
//---

   vector v2 = {(double)mlp.outputs}; //Adding the output layer to the mix of hidden layers
  
   HL_CONFIG = MatrixExtend::concatenate(HL_CONFIG, v2);
   mlp.hidden_layers = HL_CONFIG.Size();
   W_CONFIG.Resize(HL_CONFIG.Size());
     
//---

   if (y.Size() != rows)
     {
        Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
        return backprop_struct;
     }
     
     
     matrix W, B;
     
//--- GENERATE WEIGHTS
    
     this.Weights_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Bias_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Input_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Output_tensor = new CTensors((uint)mlp.hidden_layers);
     
     ulong layer_input = mlp.inputs; 
     
     for (ulong i=0; i<mlp.hidden_layers; i++)
       {
          W_CONFIG[i] = layer_input*HL_CONFIG[i];
          
          W = MatrixExtend::Random(0.0, 1.0,(ulong)HL_CONFIG[i],layer_input, m_random_state);
          
          W = W * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Weights_tensor.Add(W, i);
          
          B = MatrixExtend::Random(0.0, 0.5,(ulong)HL_CONFIG[i],1,m_random_state);
          
          B = B * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.Bias_tensor.Add(B, i);
          
          layer_input = (ulong)HL_CONFIG[i];
       }
     
//---

   if (MQLInfoInteger(MQL_DEBUG))
      Comment("<-------------------  R E G R E S S O R   N E T S  ------------------------->\n",
            "HL_CONFIG ",HL_CONFIG," TOTAL HL(S) ",mlp.hidden_layers,"\n",
            "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
            "NN INPUTS ",mlp.inputs," OUTPUT ",mlp.outputs
           );

//--- Optimizer
      
   OptimizerRMSprop optimizer_weights = optimizer;
   OptimizerRMSprop optimizer_bias = optimizer;
   
   if (batch_size>0)
    {
      OptimizerMinBGD optimizer_weights;
      OptimizerMinBGD optimizer_bias;
    }
     
//--- Cross validation

    CCrossValidation cross_validation;      
    CTensors *cv_tensor;
    matrix validation_data = MatrixExtend::concatenate(x, y);
    matrix validation_x;
    vector validation_y;
    
    cv_tensor = cross_validation.KFoldCV(validation_data, 10); //k-fold cross validation | 10 folds selected
    
//---

    matrix DELTA = {};
    double actual=0, pred=0;
    
    matrix temp_inputs ={};
    
    matrix dB = {}; //Bias Derivatives
    matrix dW = {}; //Weight Derivatives
    
   
    for (ulong epoch=0; epoch<epochs && !IsStopped(); epoch++)
      {        
        double epoch_start = GetTickCount(); 

        uint num_batches = (uint)MathFloor(x.Rows()/(batch_size+DBL_EPSILON));
        
        vector batch_loss(num_batches), 
               batch_accuracy(num_batches);
                       
         vector actual_v(1), pred_v(1), LossGradient = {};
         
         if (batch_size==0) //Stochastic Gradient Descent
          {
           for (ulong iter=0; iter<rows; iter++) //iterate through all data points
             {
               pred = predict(x.Row(iter));
               actual = y[iter];
               
               pred_v[0] = pred; 
               actual_v[0] = actual; 
   //---
                
                DELTA.Resize(mlp.outputs,1);
                
                for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                  {    
                     Partial_Derivatives = this.Output_tensor.Get(int(layer));
                     temp_inputs = this.Input_tensor.Get(int(layer));
                     
                     Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                     
                     if (mlp.hidden_layers-1 == layer) //Last layer
                      {                     
                        LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                        
                        DELTA.Col(LossGradient, 0);
                      }
                      
                     else
                      {
                        W = this.Weights_tensor.Get(layer+1);
                        
                        DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                      }
                    
                    //-- Observation | DeLTA matrix is same size as the bias matrix
                    
                    W = this.Weights_tensor.Get(layer);
                    B = this.Bias_tensor.Get(layer);
                  
                   //--- Derivatives wrt weights and bias
                  
                    dB = DELTA;
                    dW = DELTA.MatMul(temp_inputs.Transpose());                   
                    
                   //--- Weights updates
                    
                    optimizer_weights.update(W, dW);
                    optimizer_bias.update(B, dB);
                    
                    this.Weights_tensor.Add(W, layer);
                    this.Bias_tensor.Add(B, layer);
                  }
             }
         }
        else //Batch Gradient Descent
          {
               
            for (uint batch=0, batch_start=0, batch_end=batch_size; batch<num_batches; batch++, batch_start+=batch_size, batch_end=(batch_start+batch_size-1))
               {
                  matrix batch_x = MatrixExtend::Get(x, batch_start, batch_end-1);
                  vector batch_y = MatrixExtend::Get(y, batch_start, batch_end-1);
                  
                  rows = batch_x.Rows();              
                  
                    for (ulong iter=0; iter<rows ; iter++) //iterate through all data points
                      {
                        pred_v[0] = predict(batch_x.Row(iter));
                        actual_v[0] = y[iter];
                        
            //---
                        
                      DELTA.Resize(mlp.outputs,1);
                      
                      for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--) //Loop through the network backward from last to first layer
                        {    
                           Partial_Derivatives = this.Output_tensor.Get(int(layer));
                           temp_inputs = this.Input_tensor.Get(int(layer));
                           
                           Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                           
                           if (mlp.hidden_layers-1 == layer) //Last layer
                            {                     
                              LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(m_loss_function));
                              
                              DELTA.Col(LossGradient, 0);
                            }
                            
                           else
                            {
                              W = this.Weights_tensor.Get(layer+1);
                              
                              DELTA = (W.Transpose().MatMul(DELTA)) * Partial_Derivatives;
                            }
                          
                          //-- Observation | DeLTA matrix is same size as the bias matrix
                          
                          W = this.Weights_tensor.Get(layer);
                          B = this.Bias_tensor.Get(layer);
                        
                         //--- Derivatives wrt weights and bias
                        
                          dB = DELTA;
                          dW = DELTA.MatMul(temp_inputs.Transpose());                   
                          
                         //--- Weights updates
                          
                          optimizer_weights.update(W, dW);
                          optimizer_bias.update(B, dB);
                          
                          this.Weights_tensor.Add(W, layer);
                          this.Bias_tensor.Add(B, layer);
                        }
                    }
                 
                 pred_v = predict(batch_x);
                 
                 batch_loss[batch] = pred_v.Loss(batch_y, ENUM_LOSS_FUNCTION(m_loss_function));
                 batch_loss[batch] = MathIsValidNumber(batch_loss[batch]) ? (batch_loss[batch]>1e6 ? 1e6 : batch_loss[batch]) : 1e6; //Check for nan and return some large value if it is nan
                 
                 batch_accuracy[batch] = Metrics::r_squared(batch_y, pred_v);
                 
                 if (show_batch_progress)
                  printf("----> batch[%d/%d] batch-loss %.5f accuracy %.3f",batch+1,num_batches,batch_loss[batch], batch_accuracy[batch]);  
              }
          }
          
//--- End of an epoch
      
        vector validation_loss(cv_tensor.SIZE);
        vector validation_acc(cv_tensor.SIZE);
        for (ulong i=0; i<cv_tensor.SIZE; i++)
          {
            validation_data = cv_tensor.Get(i);
            MatrixExtend::XandYSplitMatrices(validation_data, validation_x, validation_y);
            
            vector val_preds = this.predict(validation_x);;
            
            validation_loss[i] = val_preds.Loss(validation_y, ENUM_LOSS_FUNCTION(m_loss_function));
            validation_acc[i] = Metrics::r_squared(validation_y, val_preds);
          }
                  
        pred_v = this.predict(x);
        
        if (batch_size==0)
          {      
              backprop_struct.training_loss[epoch] = pred_v.Loss(y, ENUM_LOSS_FUNCTION(m_loss_function));
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
        else
          {
              backprop_struct.training_loss[epoch] = batch_loss.Mean();
              backprop_struct.training_loss[epoch] = MathIsValidNumber(backprop_struct.training_loss[epoch]) ? (backprop_struct.training_loss[epoch]>1e6 ? 1e6 : backprop_struct.training_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
              backprop_struct.validation_loss[epoch] = validation_loss.Mean();
              backprop_struct.validation_loss[epoch] = MathIsValidNumber(backprop_struct.validation_loss[epoch]) ? (backprop_struct.validation_loss[epoch]>1e6 ? 1e6 : backprop_struct.validation_loss[epoch]) : 1e6; //Check for nan and return some large value if it is nan
          }
          
        double epoch_stop = GetTickCount();  
        printf("--> Epoch [%d/%d] training -> loss %.8f accuracy %.3f validation -> loss %.5f accuracy %.3f | Elapsed %s ",epoch+1,epochs,backprop_struct.training_loss[epoch],Metrics::r_squared(y, pred_v),backprop_struct.validation_loss[epoch],validation_acc.Mean(),this.ConvertTime((epoch_stop-epoch_start)/1000.0));
     }
     
   isBackProp = false;
   
  if (CheckPointer(this.Input_tensor) != POINTER_INVALID)  delete(this.Input_tensor);
  if (CheckPointer(this.Output_tensor) != POINTER_INVALID)  delete(this.Output_tensor); 
  if (CheckPointer(optimizer)!=POINTER_INVALID)  
    delete optimizer;
    
   return backprop_struct;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::fit(const matrix &x, const vector &y, OptimizerSGD *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
  trained = true; //The fit method has been called
  
  vector epochs_vector(epochs);  for (uint i=0; i<epochs; i++) epochs_vector[i] = i+1;
  
  backprop backprop_struct;
  
  backprop_struct = this.backpropagation(x, y, optimizer, epochs, batch_size, show_batch_progress); //Run backpropagation
  

  CPlots plt;
    
  backprop_struct.training_loss = log10(backprop_struct.training_loss); //Logarithmic scalling
  plt.Plot("Loss vs Epochs",epochs_vector,backprop_struct.training_loss,"epochs","optimizer-SGD log10(loss)","training-loss",CURVE_LINES);
  backprop_struct.validation_loss = log10(backprop_struct.validation_loss);
  plt.AddPlot(backprop_struct.validation_loss,"validation-loss",clrRed);
  
   while (MessageBox("Close or Cancel Loss Vs Epoch plot to proceed","Training progress",MB_OK)<0)
    Sleep(1);

  isBackProp = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::fit(const matrix &x, const vector &y, OptimizerAdaDelta *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
  trained = true; //The fit method has been called
  
  vector epochs_vector(epochs);  for (uint i=0; i<epochs; i++) epochs_vector[i] = i+1;
  
  backprop backprop_struct;
  
  backprop_struct = this.backpropagation(x, y, optimizer, epochs, batch_size, show_batch_progress); //Run backpropagation
  

  CPlots plt;
    
  backprop_struct.training_loss = log10(backprop_struct.training_loss); //Logarithmic scalling
  plt.Plot("Loss vs Epochs",epochs_vector,backprop_struct.training_loss,"epochs","optimizer-AdaDelta log10(loss)","training-loss",CURVE_LINES);
  backprop_struct.validation_loss = log10(backprop_struct.validation_loss);
  plt.AddPlot(backprop_struct.validation_loss,"validation-loss",clrRed);
  
   while (MessageBox("Close or Cancel Loss Vs Epoch plot to proceed","Training progress",MB_OK)<0)
    Sleep(1);

  isBackProp = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::fit(const matrix &x, const vector &y, OptimizerAdaGrad *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
  trained = true; //The fit method has been called
  
  vector epochs_vector(epochs);  for (uint i=0; i<epochs; i++) epochs_vector[i] = i+1;
  
  backprop backprop_struct;
  
  backprop_struct = this.backpropagation(x, y, optimizer, epochs, batch_size, show_batch_progress); //Run backpropagation
  

  CPlots plt;
    
  backprop_struct.training_loss = log10(backprop_struct.training_loss); //Logarithmic scalling
  plt.Plot("Loss vs Epochs",epochs_vector,backprop_struct.training_loss,"epochs","optimizer-AdaGrad log10(loss)","training-loss",CURVE_LINES);
  backprop_struct.validation_loss = log10(backprop_struct.validation_loss);
  plt.AddPlot(backprop_struct.validation_loss,"validation-loss",clrRed);
  
   while (MessageBox("Close or Cancel Loss Vs Epoch plot to proceed","Training progress",MB_OK)<0)
    Sleep(1);

  isBackProp = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::fit(const matrix &x, const vector &y, OptimizerAdam *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
  trained = true; //The fit method has been called
  
  vector epochs_vector(epochs);  for (uint i=0; i<epochs; i++) epochs_vector[i] = i+1;
  
  backprop backprop_struct;
  
  backprop_struct = this.backpropagation(x, y, optimizer, epochs, batch_size, show_batch_progress); //Run backpropagation
  

  CPlots plt;
    
  backprop_struct.training_loss = log10(backprop_struct.training_loss); //Logarithmic scalling
  plt.Plot("Loss vs Epochs",epochs_vector,backprop_struct.training_loss,"epochs","optimizer-Adam log10(loss)","training-loss",CURVE_LINES);
  backprop_struct.validation_loss = log10(backprop_struct.validation_loss);
  plt.AddPlot(backprop_struct.validation_loss,"validation-loss",clrRed);
  
   while (MessageBox("Close or Cancel Loss Vs Epoch plot to proceed","Training progress",MB_OK)<0)
    Sleep(1);

  isBackProp = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::fit(const matrix &x, const vector &y, OptimizerNadam *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
  trained = true; //The fit method has been called
  
  vector epochs_vector(epochs);  for (uint i=0; i<epochs; i++) epochs_vector[i] = i+1;
  
  backprop backprop_struct;
  
  backprop_struct = this.backpropagation(x, y, optimizer, epochs, batch_size, show_batch_progress); //Run backpropagation
  

  CPlots plt;
    
  backprop_struct.training_loss = log10(backprop_struct.training_loss); //Logarithmic scalling
  plt.Plot("Loss vs Epochs",epochs_vector,backprop_struct.training_loss,"epochs","optimizer-Nadam log10(loss)","training-loss",CURVE_LINES);
  backprop_struct.validation_loss = log10(backprop_struct.validation_loss);
  plt.AddPlot(backprop_struct.validation_loss,"validation-loss",clrRed);
  
   while (MessageBox("Close or Cancel Loss Vs Epoch plot to proceed","Training progress",MB_OK)<0)
    Sleep(1);

  isBackProp = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::fit(const matrix &x, const vector &y, OptimizerRMSprop *optimizer, const uint epochs, uint batch_size=0, bool show_batch_progress=false)
 {
  trained = true; //The fit method has been called
  
  vector epochs_vector(epochs);  for (uint i=0; i<epochs; i++) epochs_vector[i] = i+1;
  
  backprop backprop_struct;
  
  backprop_struct = this.backpropagation(x, y, optimizer, epochs, batch_size, show_batch_progress); //Run backpropagation
  

  CPlots plt;
    
  backprop_struct.training_loss = log10(backprop_struct.training_loss); //Logarithmic scalling
  plt.Plot("Loss vs Epochs",epochs_vector,backprop_struct.training_loss,"epochs","optimizer-RMSProp log10(loss)","training-loss",CURVE_LINES);
  backprop_struct.validation_loss = log10(backprop_struct.validation_loss);
  plt.AddPlot(backprop_struct.validation_loss,"validation-loss",clrRed);
  
   while (MessageBox("Close or Cancel Loss Vs Epoch plot to proceed","Training progress",MB_OK)<0)
    Sleep(1);

  isBackProp = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+