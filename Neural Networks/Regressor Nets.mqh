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

#ifndef RANDOM_STATE
#define  RANDOM_STATE 42
#endif 

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

enum optimizer
 {
   OPTIMIZER_ADAM,
   OPTIMIZER_NONE
 };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum norm
 {
   NORM_L1, //L1Normalization
   NORM_L2  //L2 normalization
 };
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
    LOSS_MSE_ = LOSS_MSE,
    LOSS_MAE_ = LOSS_MAE,
    LOSS_MSLE_ = LOSS_MSLE,
    LOSS_HUBER_ = LOSS_HUBER
  };

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
   
   CTensors          *W_tensor; //Weight Tensor
   CTensors          *B_tensor;
   CTensors          *Input_tensor;
   CTensors          *Output_tensor;
   CTensors          *Derivatives_tensor_WRTW;
   CTensors          *Derivatives_tensor_WRTB;
   
protected:
   activation        A_FX;

   string            CalcTimeElapsed(double seconds);

protected:
   matrix            W_MATRIX; //Weights Matrix
   matrix            B_MATRIX; //Bias Matrix
   vector            W_CONFIG;
   vector            HL_CONFIG; 
   
private: //for backpropn
   bool              isBackProp; 
   matrix<double>    ACTIVATIONS;
   matrix<double>    Partial_Derivatives; 
   
   matrix Sign(matrix &W);
   matrix L1_Normalization(double lambda, matrix &W);
   matrix L2_Normalization(double lambda, matrix &W);
   
private: //Optimizers

   CTensors *mw_tensor; //moment weight
   CTensors *vmw_tensor; //vector moment weight
   
   CTensors *mb_tensor; //moment bias
   CTensors *vmb_tensor; //vector moment bias
   
   void  AdamOptimizerW(matrix &dx_W, matrix &moment, matrix &v_matrix, matrix &moment_hat, matrix &v_matrix_hat, matrix &W, const double alpha=0.01, const double beta1=0.9, const double beta2=0.999, const double epsilon=1e-8);
   void  AdamOptimizerB(matrix &dx_B, matrix &moment, matrix &v_matrix, matrix &moment_hat, matrix &v_matrix_hat, matrix &B,int time_step, const double alpha=0.01, const double beta1=0.9, const double beta2=0.999, const double epsilon=1e-8);
   
   void  RegressorNetsBackProp(matrix& x, vector &y, uint epochs, double alpha, loss LossFx=LOSS_MSE_, optimizer OPTIMIZER=OPTIMIZER_ADAM);
   
public:

                     CRegressorNets(vector &HL_NODES, activation ActivationFX=AF_RELU_);
                    ~CRegressorNets(void);

   void fit(matrix &x, vector &y);
   double predict(vector &x);
   vector predict(matrix &x);
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRegressorNets::CRegressorNets(vector &HL_NODES, activation ActivationFX=AF_RELU_)
 :A_FX(ActivationFX),
  isBackProp(false)
  {   
     HL_CONFIG.Copy(HL_NODES);
     
     vector v2 = {(double)mlp.outputs};
     
     HL_CONFIG = MatrixExtend::concatenate(HL_CONFIG, v2);
     
//---

     W_CONFIG.Resize(HL_CONFIG.Size()); 
     mlp.hidden_layers = HL_CONFIG.Size();
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRegressorNets::~CRegressorNets(void)
  {
   delete(this.W_tensor);
   delete(this.B_tensor);
   delete(this.Input_tensor);
   delete(this.Output_tensor);
   delete(this.Derivatives_tensor_WRTW);
   delete(this.Derivatives_tensor_WRTB);

//--- Adam optimizer

   delete(mw_tensor); 
   delete(vmw_tensor); 
   delete(mb_tensor);
   delete(vmb_tensor);
   
   ZeroMemory(W_CONFIG);
   ZeroMemory(HL_CONFIG); 
   
   isBackProp = false; 
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CRegressorNets::predict(vector &x)
  {   
   matrix L_INPUT = MatrixExtend::VectorToMatrix(x); 
   
   matrix L_OUTPUT ={};
    
   for(ulong i=0; i<mlp.hidden_layers; i++)
     { 
      this.W_MATRIX = this.W_tensor.Get(i);
      this.B_MATRIX = this.B_tensor.Get(i);
     
      if (isBackProp)  this.Input_tensor.Add(L_INPUT, i); 
      
      L_OUTPUT = W_MATRIX.MatMul(L_INPUT); //W x I

//--- 

      L_OUTPUT = L_OUTPUT+B_MATRIX; //Add bias
      L_OUTPUT.Activation(L_OUTPUT, ENUM_ACTIVATION_FUNCTION(A_FX)); //Activation
      
      L_INPUT = L_OUTPUT;
      
      if (isBackProp)  this.Output_tensor.Add(L_OUTPUT, i);
     }

//---
   
   vector out = MatrixExtend::MatrixToVector(L_OUTPUT);
   
   #ifdef DEBUG_MODE 
      //Print("outputs\n ",L_OUTPUT);
   #endif 
   
    return(out[0]);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CRegressorNets::predict(matrix &x)
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

string CRegressorNets::CalcTimeElapsed(double seconds)
  {
   string time_str = "";

   uint minutes=0, hours=0;

   if(seconds >= 60)
      time_str = StringFormat("%d Minutes and %.3f Seconds ",minutes=(int)round(seconds/60.0), ((int)seconds % 60));
   if(minutes >= 60)
      time_str = StringFormat("%d Hours %d Minutes and %.3f Seconds ",hours=(int)round(minutes/60.0), minutes, ((int)seconds % 60));
   else
      time_str = StringFormat("%.3f Seconds ",seconds);

   return time_str;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::RegressorNetsBackProp(matrix& x, vector &y, uint epochs, double alpha, loss LossFx=LOSS_MSE_, optimizer OPTIMIZER=OPTIMIZER_ADAM)
 {
    isBackProp = true;

   matrix XMatrix = x; 
   
//---

   ulong rows = x.Rows();
   
   mlp.inputs = x.Cols();
   mlp.outputs = 1;
   
   if (y.Size() != rows)
     {
        Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
        return;
     }
     
//--- GENERATE WEIGHTS
    
     this.W_tensor = new CTensors((uint)mlp.hidden_layers);
     this.B_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Input_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Output_tensor = new CTensors((uint)mlp.hidden_layers);
     this.Derivatives_tensor_WRTW = new CTensors((uint)mlp.hidden_layers);
     this.Derivatives_tensor_WRTB = new CTensors((uint)mlp.hidden_layers);
     
     ulong layer_input = mlp.inputs; 
     
     for (ulong i=0; i<mlp.hidden_layers; i++)
       {
          W_CONFIG[i] = layer_input*HL_CONFIG[i];
          
          W_MATRIX = MatrixExtend::Random(0.0, 1.0,(ulong)HL_CONFIG[i],layer_input, RANDOM_STATE);
          
          this.W_MATRIX = this.W_MATRIX * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.W_tensor.Add(W_MATRIX, i);
          
          B_MATRIX = MatrixExtend::Random(0.0, 0.5,(ulong)HL_CONFIG[i],1,RANDOM_STATE);
          
          this.B_MATRIX = this.B_MATRIX * sqrt(2/((double)layer_input + HL_CONFIG[i])); //glorot
          this.B_tensor.Add(B_MATRIX, i);
          
          layer_input = (ulong)HL_CONFIG[i];
       }
     
//---

   #ifdef  DEBUG_MODE
   
      Comment("<-------------------  R E G R E S S O R   N E T S  ------------------------->\n",
            "HL_CONFIG ",HL_CONFIG," TOTAL HL(S) ",mlp.hidden_layers,"\n",
            "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
            "NN INPUTS ",inputs," OUTPUT ",outputs," NORMALIZATION ",EnumToString(NORM_METHOD)
           );
    
   #endif

//--- for adam 
     
    matrix temp_w, temp_b;
    
    matrix moment_w, moment_hat_w, v_matrix_w, v_matrix_hat_w;
    matrix moment_b, moment_hat_b, v_matrix_b, v_matrix_hat_b;  
    
    switch(OPTIMIZER)
      {
       case  OPTIMIZER_ADAM:
          mw_tensor = new CTensors((uint)mlp.hidden_layers); 
          vmw_tensor = new CTensors((uint)mlp.hidden_layers); 
          
          mb_tensor = new CTensors((uint)mlp.hidden_layers);
          vmb_tensor = new CTensors((uint)mlp.hidden_layers);
          
          for (ulong i=0; i<mlp.hidden_layers; i++)
            {
               temp_w = this.W_tensor.Get(i);
               moment_w.Resize(temp_w.Rows(), temp_w.Cols()); moment_hat_w.Resize(temp_w.Rows(), temp_w.Cols()); v_matrix_w.Resize(temp_w.Rows(), temp_w.Cols()); v_matrix_hat_w.Resize(temp_w.Rows(), temp_w.Cols());
               
               temp_b = this.B_tensor.Get(i);               
               moment_b.Resize(temp_b.Rows(), temp_b.Cols()); moment_hat_b.Resize(temp_b.Rows(), temp_b.Cols()); v_matrix_b.Resize(temp_b.Rows(), temp_b.Cols()); v_matrix_hat_b.Resize(temp_b.Rows(), temp_b.Cols());
   
               moment_w.Fill(0); moment_hat_w.Fill(0); v_matrix_w.Fill(0); v_matrix_hat_w.Fill(0);   
               moment_b.Fill(0); moment_hat_b.Fill(0); v_matrix_b.Fill(0); v_matrix_hat_b.Fill(0); 
               
               mw_tensor.Add(moment_w, i);
               vmw_tensor.Add(v_matrix_w, i);
               
               mb_tensor.Add(moment_b, i);
               vmb_tensor.Add(v_matrix_b, i);
            }
         break;
       default:
         break;
      }
//---

    matrix DELTA(1,1), temp_delta={};
    double actual=0, pred=0;
    
    matrix temp_derivatives(1,1), temp_inputs ={}, temp_outputs ={};
    temp_derivatives.Fill(0);
    matrix derivatives_cache ={};
    
    matrix B_DX = {}; //Bias Derivatives
    matrix W_DX = {}; //Weight Derivatives
    
    vector actual_v ={actual}, pred_v={0}, LossGradient = {};
   
    for (ulong epoch=0; epoch<epochs && !IsStopped(); epoch++)
      { 
       vector actuals(rows);
       vector preds(rows); 
       
        double epoch_start = GetMicrosecondCount()/(double)1e6; 
        
        for (ulong iter=0; iter<rows; iter++) //replace to rows
          {
            pred = predict(XMatrix.Row(iter));
            actual = y[iter];
            
            pred_v[0] = pred; preds[iter] = pred;
            actual_v[0] = actual; actuals[iter] = actual;
//---

            //Print("Iteration ",iter+1);
             
             DELTA.Resize(1,1);
             
             for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--)
               {    
                                
                  temp_outputs = this.Output_tensor.Get(layer);
                  Partial_Derivatives = this.Output_tensor.Get(int(layer));
                  temp_inputs = this.Input_tensor.Get(int(layer));
                  
                  this.B_MATRIX = this.B_tensor.Get(layer);
                  
                  Partial_Derivatives.Derivative(Partial_Derivatives, ENUM_ACTIVATION_FUNCTION(A_FX));
                  
                  if (mlp.hidden_layers-1 == layer) //Last layer
                   {                     
                     LossGradient = pred_v.LossGradient(actual_v, ENUM_LOSS_FUNCTION(LossFx));
                     
                     DELTA.Col(LossGradient, 0);
                   }
                   
                  else
                   {
                     
                     this.W_MATRIX = this.W_tensor.Get(layer+1);
                     
                     this.W_MATRIX = this.W_MATRIX.Transpose();
                     
                     temp_delta = this.W_MATRIX.MatMul(DELTA);
                     
                     DELTA = temp_delta * Partial_Derivatives;
                     
                     this.W_MATRIX = this.W_MATRIX.Transpose();
                     
                   }
                 
                 //-- Observation | DeLTA matrix is same size as the bias matrix
                 
                 this.W_MATRIX = this.W_tensor.Get(layer);
                 this.B_MATRIX = this.B_tensor.Get(layer);
                 
                 B_DX = DELTA;
                 this.Derivatives_tensor_WRTB.Add(B_DX, layer);
                 
                 temp_inputs = temp_inputs.Transpose();
                 W_DX = DELTA.MatMul(temp_inputs);  
                 this.Derivatives_tensor_WRTW.Add(W_DX, layer);
                 
                 //Print(" w_dx\n ",W_DX);
               }

//--- SGD & ohter Optimizers | updating weights procedure
         
          //Print("W before ",this.W_MATRIX);
 
           for (int layer=(int)mlp.hidden_layers-1; layer>=0 && !IsStopped(); layer--)
            {
              this.W_MATRIX = this.W_tensor.Get(layer);
              this.B_MATRIX = this.B_tensor.Get(layer);
              
              W_DX = Derivatives_tensor_WRTW.Get(layer);
              B_DX = Derivatives_tensor_WRTB.Get(layer);
              
              switch(OPTIMIZER)
                {
                 case OPTIMIZER_ADAM:
                   
                   moment_w = mw_tensor.Get(layer);
                   v_matrix_w = vmw_tensor.Get(layer);
                   
                   //Print("moment_w\n",moment_w,"\nv_matrix\n",v_matrix_w,"\nDerivatives weights\n",W_DX);
                   
                   AdamOptimizerW(W_DX, moment_w, v_matrix_w, moment_hat_w, v_matrix_hat_w ,this.W_MATRIX);
                   
                   //Print("Weights updates\n",this.W_MATRIX);
                   
                   mw_tensor.Add(moment_w, layer);
                   vmw_tensor.Add(v_matrix_w, layer);
                     
                 //---
                 
                   moment_b = mb_tensor.Get(layer);
                   v_matrix_b = vmb_tensor.Get(layer);
                   
                   AdamOptimizerB(B_DX, moment_b,v_matrix_b, moment_hat_b, v_matrix_hat_w, this.B_MATRIX, (int)iter);
                   
                   mb_tensor.Add(moment_b, layer);
                   vmb_tensor.Add(v_matrix_b, layer);
                   
                   break;
                 case OPTIMIZER_NONE:
                 
                    this.W_MATRIX = this.W_MATRIX - (alpha * W_DX);
                    this.B_MATRIX = this.B_MATRIX - (alpha * B_DX);
                    
                   break;
                }
              
              //Print("Weight after ",this.W_MATRIX);
              
              this.W_tensor.Add(this.W_MATRIX, layer);
              this.B_tensor.Add(this.B_MATRIX, layer);
            }
         }
       
        double epoch_stop = GetMicrosecondCount()/(double)1e6;
         
        printf("[ Epoch %d/%d ] Cost %.8f Acc %.8f | Elapsed %s ",epoch+1,epochs,preds.Loss(actuals, ENUM_LOSS_FUNCTION(LossFx)),Metrics::r_squared(actuals, preds),this.CalcTimeElapsed(epoch_stop-epoch_start));
     }
     
   isBackProp = false;
//---
   
   Input_tensor.MemoryClear();
   Output_tensor.MemoryClear();
   Derivatives_tensor_WRTW.MemoryClear();
   Derivatives_tensor_WRTB.MemoryClear()
   ;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::AdamOptimizerW(matrix &dx_W,
                                  matrix &moment,
                                  matrix &v_matrix,
                                  matrix &moment_hat, 
                                  matrix &v_matrix_hat, 
                                  matrix &W, 
                                  const double alpha=0.01,
                                  const double beta1=0.9, 
                                  const double beta2=0.999, 
                                  const double epsilon=1e-8)
{ 
    //printf("moment[%dx%d] dx_w[%dx%d]",moment.Rows(), moment.Cols(), dx_W.Rows(), dx_W.Cols());
    
    //Print("DX_W\n",dx_W,"\nW\n",W);
    
    moment = (beta1 * moment) + ((1 - beta1) * dx_W);
    v_matrix = (beta2 * v_matrix) + ((1-beta2) * MathPow(dx_W, 2));
    
    //Print("moment\n",moment,"\nv_matrix\n",v_matrix);
    
    moment_hat   = moment   / (1 - beta1);
    v_matrix_hat = v_matrix / (1 - beta2);
    
    //Print("moment hat\n",moment_hat,"\nv_matrix hat\n",v_matrix_hat);
    
    //Print("\nInside ---> Weight before\n",W,"\nv_matrix_hat\n",v_matrix_hat,"\nmoment hat\n",moment_hat);
    
    //Print("All the maths ",( (m_alpha * moment_hat) / (MathSqrt(v_matrix_hat) + epsilon) ));
    
    W -= ( (alpha * moment_hat) / (MathSqrt(v_matrix_hat) + epsilon) );
    
    //Print("w update\n",W);
    //Print("\nInside ---> Weight after\n",W);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRegressorNets::AdamOptimizerB(matrix &dx_B,
                                   matrix &moment,
                                   matrix &v_matrix,
                                   matrix &moment_hat,
                                   matrix &v_matrix_hat,
                                   matrix &B,
                                   int time_step,
                                   const double alpha=0.01,
                                   const double beta1=0.9,
                                   const double beta2=0.999,
                                   const double epsilon=1e-8)
{
   //Print("DX_B\n",dx_B,"\nB\n",B,"\nBeta1 ",beta1," beta2 ",beta2," epsilon ",epsilon," LR ",m_alpha);
   
    moment = (beta1 * moment) + ((1 - beta1) * dx_B);
    v_matrix = (beta2 * v_matrix) + ((1 - beta2) * MathPow(dx_B, 2));
    
    //Print("moment\n",moment,"\nv_matrix\n",v_matrix);
    
    moment_hat   = moment   / (1 - MathPow(beta1, time_step));
    v_matrix_hat = v_matrix / (1 - MathPow(beta2, time_step));
    
    //Print("moment hat\n",moment_hat,"\nv_matrix hat\n",v_matrix_hat);
    
    B -= ( (alpha * moment_hat) / (MathSqrt(v_matrix_hat) + epsilon) );
    
    //Print("Bias update\n",B);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CRegressorNets::Sign(matrix &W)
 {
   for (ulong i=0; i<W.Rows(); i++)
      for (ulong j=0; j<W.Cols(); j++)
         {
            if (W[i][j] < 0)
               W[i][j] = -1;
            else if (W[i][j] == 0)
               W[i][j] = 0;
            else 
               W[i][j] = 1;
         }
     return (W);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

matrix CRegressorNets::L1_Normalization(double lambda, matrix &W)
 {
   return (lambda * Sign(W));
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CRegressorNets::L2_Normalization(double lambda, matrix &W)
 {
   
   return (2* lambda * W);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
