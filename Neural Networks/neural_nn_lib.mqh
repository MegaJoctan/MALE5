    //+------------------------------------------------------------------+
//|                                                neural_nn_lib.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
//| Regression NN class 

#include <MALE5\preprocessing.mqh>;
#include <MALE5\matrix_utils.mqh>;
#include <MALE5\metrics.mqh>

#ifndef RANDOM_STATE 
 #define  RANDOM_STATE 42
 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CNeuralNets
  {
   CPreprocessing pre_processing;
   CMatrixutils   matrix_utils;
   CMetrics       metrics;
   
   protected:
                  bool                     SoftMaxLayer;
                  ENUM_ACTIVATION_FUNCTION A_FX;                  
                  matrix                   XMatrix;
                  vector                   YVector;
                  
                  ulong                    m_hLayers;
                  vector                   m_targetVector;
                  vector                   classes;
                  
                  ulong                    m_rows, m_cols;
                  ulong                    m_inputs;
                  uint                     m_outputs;  //MLP outputs
   
                  string   CalcTimeElapsed(double seconds);
                  bool    isNNClassification();
                  
   protected:
                  vector W; //Weights vector
                  vector B; //Bias vector 
                  vector W_CONFIG; 
                  vector HL_CONFIG;
                  vector IN_CONFIG; //All the inputs from hidden layer to the next layer
                  
                  void   SoftMaxLayerFX(matrix<double> &mat);
                                    
   public:
                  
                  CNeuralNets(matrix &xmatrix, vector &yvector, ENUM_ACTIVATION_FUNCTION ActivationFX, vector &HL_CONFIG, bool SoftMaxLyr=false);
                 ~CNeuralNets(void);
                  
                  vector  FeedForwardMLP(vector& V_in);  
                           
  };

//+------------------------------------------------------------------+

CNeuralNets::CNeuralNets(matrix &xmatrix, vector &yvector, ENUM_ACTIVATION_FUNCTION ActivationFX, vector &HL_CONFIG_, bool SoftMaxLyr=false)
 {
 
    A_FX = ActivationFX; 

//--
     
    XMatrix = xmatrix;
    YVector = yvector;

    SoftMaxLayer = SoftMaxLyr;
    

    m_inputs = XMatrix.Cols();
    m_rows = XMatrix.Rows();
    m_cols = XMatrix.Cols(); 

   
//--- Decide the outputs of NN
   
    if (A_FX == AF_SIGMOID || SoftMaxLayer)
       {
          classes = matrix_utils.Classes(YVector);
          m_outputs = (uint)classes.Size();
       }
       
    else  m_outputs = 1;

//---

    HL_CONFIG.Copy(HL_CONFIG_);
    
    IN_CONFIG.Resize(1);
    IN_CONFIG[0] = int(m_inputs);

    IN_CONFIG =matrix_utils.Append(IN_CONFIG,HL_CONFIG);
    
//---

    HL_CONFIG.Resize(HL_CONFIG.Size()+1); //Add the output layer
    HL_CONFIG[HL_CONFIG.Size()-1] = m_outputs; //Append one node to the output layer
    
    B.Resize((ulong)HL_CONFIG.Sum());
    
    m_hLayers = HL_CONFIG.Size();    
   
//--- GENERATE WEIGHTS
   
    vector v(HL_CONFIG.Size());
  
    ulong inputs = m_inputs; 
    
    for (ulong i=0; i<v.Size(); i++)
      {
         v[i] = inputs*HL_CONFIG[i];
         inputs = (ulong)HL_CONFIG[i];
      }
     
     W_CONFIG = v;
     
     W = matrix_utils.Random(0.0, 1.0, (int)v.Sum(),RANDOM_STATE);
    
//--- GENERATE BIAS
    
     B = matrix_utils.Random(0.0,1.0,(int)B.Size(),RANDOM_STATE);
   
      #ifdef  DEBUG_MODE 
    
            Print( "<------------------- NN INFO  ------------------------->\n",
                      "HL_CONFIG ",HL_CONFIG," WEIGHTS ",W.Size(),"\n","TOTAL HL(S) ",m_hLayers,"\n",
                      "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
                      "NN INPUTS ",m_inputs," OUTPUT ",m_outputs,"\n",
                      "IN_CONFIG ",IN_CONFIG," Softmax Layer ",bool(SoftMaxLayer),"\n",
                      "BIAS ",B,"\n",
                      "--------------------        ------------------------->"
                 );
            
      #endif      

 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

CNeuralNets::~CNeuralNets(void)
 {
   ZeroMemory(XMatrix);
   ZeroMemory(YVector);
   
   SoftMaxLayer = false;
   
   ZeroMemory(W);
   ZeroMemory(B); 
   ZeroMemory(W_CONFIG); 
   ZeroMemory(HL_CONFIG);
   ZeroMemory(IN_CONFIG);
   
   ZeroMemory(XMatrix);
   ZeroMemory(YVector);
    
   ZeroMemory(m_targetVector);
   ZeroMemory(classes);
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

vector CNeuralNets::FeedForwardMLP(vector &V_in)
 {    
   matrix L_INPUT = {}, L_OUTPUT={}, L_WEIGHTS = {};
   vector v_weights ={};
   
   ulong start = 0;             
   
   L_INPUT = matrix_utils.VectorToMatrix(V_in); 
   
   vector L_BIAS_VECTOR = {};
   matrix L_BIAS_MATRIX = {};
   
   ulong b_start = 0;
   
   for (ulong i=0; i<W_CONFIG.Size(); i++)
      {
         Print("--? ",i);
         
         matrix_utils.Copy(W,v_weights,start,ulong(W_CONFIG[i]));
         
         L_WEIGHTS = matrix_utils.VectorToMatrix(v_weights,L_INPUT.Rows());
         
         matrix_utils.Copy(B,L_BIAS_VECTOR,b_start,(ulong)HL_CONFIG[i]);
         L_BIAS_MATRIX = matrix_utils.VectorToMatrix(L_BIAS_VECTOR);
         
         Print("L_WEIGHTS\n",L_WEIGHTS,"\nL_INPUT\n",L_INPUT,"\nL_BIAS\n",L_BIAS_MATRIX);
         
         L_OUTPUT = L_WEIGHTS.MatMul(L_INPUT);

         L_OUTPUT = L_OUTPUT+L_BIAS_MATRIX; //Add bias

//---
         
         if (i==W_CONFIG.Size()-1) //Last layer
          {
             if (SoftMaxLayer)  
               SoftMaxLayerFX(L_OUTPUT);
             else
               L_OUTPUT.Activation(L_OUTPUT, A_FX);
          }
         else
            L_OUTPUT.Activation(L_OUTPUT, A_FX);
            
//---

         L_INPUT.Copy(L_OUTPUT); //Assign outputs to the inputs
         start += (ulong)W_CONFIG[i]; //New weights copy
         b_start += (ulong)HL_CONFIG[i];
         
      }
   Print("outputs\n ",L_OUTPUT);
   
   return(matrix_utils.MatrixToVector(L_OUTPUT));
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CNeuralNets::SoftMaxLayerFX(matrix<double> &mat)
 {
   vector<double> ret = matrix_utils.MatrixToVector(mat);
   
   ret.Activation(ret, AF_SOFTMAX);
   
   mat = matrix_utils.VectorToMatrix(ret, mat.Cols());
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CNeuralNets::isNNClassification(void)
 { 
    switch(A_FX)
      {
       case  AF_ELU:           return false;          break;  //Exponential Linear Unit
       case  AF_EXP:           return false;          break;  //Exponential
       case  AF_GELU:          return false;          break;  //Gaussian Error Linear Unit
       case AF_LINEAR:         return false;          break;  //Linear
       case AF_LRELU:          return false;          break;  //Leaky Rectified linear unit
       case AF_RELU:           return false;          break;  //Rectified linear unit
       case AF_SELU:           return false;          break;  //Scaled exponential linear unit
       case AF_TRELU:          return false;          break;  //Threshold Rectified linear unit
       case AF_SOFTPLUS:       return false;          break;  //Softplus
       
       case AF_HARD_SIGMOID:  return true;          break;  //Hard Sigmoid 
       case AF_SIGMOID:       return true;          break;  //Sigmoid
       case AF_SWISH:         return true;          break;  //Swish
       case AF_SOFTSIGN:      return true;          break;  //Softsign
       case AF_TANH:          return true;          break;  //The hyperbolic tangent Function
       
       default:
         Print("Unknown Activation Function");
         return false;
         break;
      }
      
  return(false); 
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

string CNeuralNets::CalcTimeElapsed(double seconds)
 {
  string time_str = "";
  
  uint minutes=0, hours=0;
  
   if (seconds >= 60)
     time_str = StringFormat("%d Minutes and %.3f Seconds ",minutes=(int)round(seconds/60.0), ((int)seconds % 60));     
   if (minutes >= 60)
     time_str = StringFormat("%d Hours %d Minutes and %.3f Seconds ",hours=(int)round(minutes/60.0), minutes, ((int)seconds % 60));
   else
     time_str = StringFormat("%.3f Seconds ",seconds);
     
   return time_str;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+