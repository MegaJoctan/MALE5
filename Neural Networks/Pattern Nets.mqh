//+------------------------------------------------------------------+
//|                                                 Pattern Nets.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| Neural network type for pattern recognition/ can be used to      |
//| to predict discrete data target variables. They are widely known |
//| as classification Neural Networks                                |
//+------------------------------------------------------------------+
#include <MALE5\matrix_utils.mqh>
#include <MALE5\preprocessing.mqh>

#ifndef RANDOM_STATE 
 #define  RANDOM_STATE 42
#endif 

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum scaler
 {
   MIN_MAX_SCALER,
   MEAN_NORM,
   STANDARDIZATION
 };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum activation
  {
   AF_HARD_SIGMOID_ = AF_HARD_SIGMOID,
   AF_SIGMOID_ = AF_SIGMOID,
   AF_SWISH_ = AF_SWISH,
   AF_SOFTSIGN_ = AF_SOFTSIGN,
   AF_TANH_ = AF_TANH
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CPatternNets
  {
private:
   CMatrixutils      matrix_utils;
   CPreprocessing    pre_processing;
   
   vector W_CONFIG;
   vector W; //Weights vector
   vector B; //Bias vector 
   activation  A_FX;
   scaler      NORM_ENUM;
   
protected:
   ulong    inputs;
   ulong    outputs;
   ulong    rows;
   vector   HL_CONFIG;
   bool     SoftMaxLayer;
   vector   classes;
   void     SoftMaxLayerFX(matrix<double> &mat);
   
public:
                     CPatternNets(matrix &xmatrix, vector &yvector,vector &HL_NODES, activation ActivationFx, scaler SCALER, bool SoftMaxLyr=false);
                    ~CPatternNets(void);
                    
                     int  PatternNetFF(vector &in_vector);
                     vector PatternNetFF(matrix &xmatrix); 
                     
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPatternNets::CPatternNets(matrix &xmatrix, vector &yvector,vector &HL_NODES, activation ActivationFx, scaler SCALER, bool SoftMaxLyr=false)
  {
      A_FX = ActivationFx;
      inputs = xmatrix.Cols();
      rows = xmatrix.Rows();
      SoftMaxLayer = SoftMaxLyr;
      
//--- Normalize data
      
     NORM_ENUM = SCALER;
      
     switch(NORM_ENUM)
      {
       case MIN_MAX_SCALER:
            pre_processing.MinMaxScaler(xmatrix);
         break;
       case MEAN_NORM:
            pre_processing.MeanNormalization(xmatrix);
          break;
       case STANDARDIZATION:
            pre_processing.Standardization(xmatrix);
          break;
      }  

//---

      if (rows != yvector.Size())
        {
          Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
          return;
        }
     
     classes = matrix_utils.Classes(yvector);
     outputs = classes.Size();
     
     HL_CONFIG.Copy(HL_NODES);
      
     HL_CONFIG.Resize(HL_CONFIG.Size()+1); //Add the output layer
     HL_CONFIG[HL_CONFIG.Size()-1] = (int)outputs; //Append one node to the output layer
//---
     W_CONFIG.Resize(HL_CONFIG.Size());
     B.Resize((ulong)HL_CONFIG.Sum());
     
//--- GENERATE WEIGHTS
   
     ulong layer_input = inputs; 
       
     for (ulong i=0; i<HL_CONFIG.Size(); i++)
       {
          W_CONFIG[i] = layer_input*HL_CONFIG[i];
          layer_input = (ulong)HL_CONFIG[i];
       }
     
     W.Resize((ulong)W_CONFIG.Sum());
     
     W = matrix_utils.Random(0.0, 1.0, (int)W.Size(),RANDOM_STATE); //Gen weights
     B = matrix_utils.Random(0.0,1.0,(int)B.Size(),RANDOM_STATE); //Gen bias
      
//---
     
     #ifdef DEBUG_MODE
       Comment(
                "< - - -  P A T T E R N    N E T S  - - - >\n",
                "HIDDEN LAYERS + OUTPUT ",HL_CONFIG,"\n",   
                "INPUTS ",inputs," | OUTPUTS ",outputs," W CONFIG ",W_CONFIG,"\n",
                "activation ",EnumToString(A_FX)," SoftMaxLayer = ",bool(SoftMaxLayer)
              );
              
       Print("WEIGHTS ",W,"\nBIAS ",B);
     #endif 
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPatternNets::~CPatternNets(void)
  {
    ZeroMemory(W);
    ZeroMemory(B);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CPatternNets::PatternNetFF(vector &in_vector)
 {
  
   matrix L_INPUT = {}, L_OUTPUT={}, L_WEIGHTS = {};
   vector v_weights ={};
   
   ulong w_start = 0;             
   
   L_INPUT = matrix_utils.VectorToMatrix(in_vector); 
   
   vector L_BIAS_VECTOR = {};
   matrix L_BIAS_MATRIX = {};
   
   ulong b_start = 0;
   
   for (ulong i=0; i<W_CONFIG.Size(); i++)
      {         
         matrix_utils.Copy(W,v_weights,w_start,ulong(W_CONFIG[i]));
         
         L_WEIGHTS = matrix_utils.VectorToMatrix(v_weights,L_INPUT.Rows());
         
         matrix_utils.Copy(B,L_BIAS_VECTOR,b_start,(ulong)HL_CONFIG[i]);
         L_BIAS_MATRIX = matrix_utils.VectorToMatrix(L_BIAS_VECTOR);
         
         #ifdef DEBUG_MODE
           Print("--> ",i);
           Print("L_WEIGHTS\n",L_WEIGHTS,"\nL_INPUT\n",L_INPUT,"\nL_BIAS\n",L_BIAS_MATRIX);
         #endif 
         
         L_OUTPUT = L_WEIGHTS.MatMul(L_INPUT);

         L_OUTPUT = L_OUTPUT+L_BIAS_MATRIX; //Add bias

//---
         
         if (i==W_CONFIG.Size()-1) //Last layer
          {
             if (SoftMaxLayer)  
              {
                Print("Before softmax\n",L_OUTPUT);
                SoftMaxLayerFX(L_OUTPUT);
                Print("After\n",L_OUTPUT);
              }
             else
               L_OUTPUT.Activation(L_OUTPUT, ENUM_ACTIVATION_FUNCTION(A_FX));
          }
         else
            L_OUTPUT.Activation(L_OUTPUT, ENUM_ACTIVATION_FUNCTION(A_FX));
            
//---

         L_INPUT.Copy(L_OUTPUT); //Assign outputs to the inputs
         w_start += (ulong)W_CONFIG[i]; //New weights copy
         b_start += (ulong)HL_CONFIG[i];
         
      }
   
   #ifdef DEBUG_MODE 
     Print("--> outputs\n ",L_OUTPUT);
   #endif 
   
   vector v_out = matrix_utils.MatrixToVector(L_OUTPUT);
   
   return((int)classes[v_out.ArgMax()]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

void CPatternNets::SoftMaxLayerFX(matrix<double> &mat)
 {
   vector<double> ret = matrix_utils.MatrixToVector(mat);
   
   ret.Activation(ret, AF_SOFTMAX);
   
   mat = matrix_utils.VectorToMatrix(ret, mat.Cols());
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

vector CPatternNets::PatternNetFF(matrix &xmatrix)
 {
   vector v(xmatrix.Rows());
   
    for (ulong i=0; i<xmatrix.Rows(); i++)
         v[i] = PatternNetFF(xmatrix.Row(i));      
   
   return (v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
