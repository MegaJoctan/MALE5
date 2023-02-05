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
#include <MALE5\matrix_utils.mqh>;
#include <MALE5\metrics.mqh>

#ifndef RANDOM_STATE
#define  RANDOM_STATE 42

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
class CRegressorNets
  {
   CPreprocessing    pre_processing;
   CMatrixutils      matrix_utils;
   CMetrics          metrics;

protected:
   activation        A_FX;
   matrix            XMatrix;
   vector            YVector;

   ulong             m_hLayers;
   vector            m_targetVector;
   vector            classes;

   ulong             m_rows, m_cols;
   ulong             inputs;
   uint              outputs;  //MLP outputs

   string            CalcTimeElapsed(double seconds);

protected:
   vector            W; //Weights vector
   vector            B; //Bias vector
   vector            W_CONFIG;
   vector            HL_CONFIG; 

   void              SoftMaxLayerFX(matrix<double> &mat);

public:

                     CRegressorNets(matrix &xmatrix, vector &yvector, activation ActivationFX, vector &HL_CONFIG);
                    ~CRegressorNets(void);

   double            RegressorNetsFF(vector& V_in);

  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

CRegressorNets::CRegressorNets(matrix &xmatrix, vector &yvector, activation ActivationFX, vector &HL_NODES)
  {

   A_FX = ActivationFX;

//--

   XMatrix = xmatrix;
   YVector = yvector;

   m_rows = XMatrix.Rows();
   inputs = XMatrix.Cols();

   outputs = 1;
   
   if (yvector.Size() != m_rows)
     {
        Print(__FUNCTION__," FATAL | Number of rows in the x matrix is not the same the y vector size ");
        return;
     }
     

//--- 
     
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
     B = matrix_utils.Random(0.0, 0.5,(int)B.Size(),RANDOM_STATE); //Gen bias
      
//---

   #ifdef  DEBUG_MODE
      Comment("");
      Comment("<-------------------  R E G R E S S O R   N E T S  ------------------------->\n",
            "HL_CONFIG ",HL_CONFIG," WEIGHTS ",W.Size(),"\n","TOTAL HL(S) ",m_hLayers,"\n",
            "W_CONFIG ",W_CONFIG," ACTIVATION ",EnumToString(A_FX),"\n",
            "NN INPUTS ",inputs," OUTPUT ",outputs,"\n", 
            "BIAS ",B
           );
   
   #endif

  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
 
CRegressorNets::~CRegressorNets(void)
  {
   ZeroMemory(XMatrix);
   ZeroMemory(YVector);

   ZeroMemory(W);
   ZeroMemory(B);
   ZeroMemory(W_CONFIG);
   ZeroMemory(HL_CONFIG); 

   ZeroMemory(XMatrix);
   ZeroMemory(YVector);

   ZeroMemory(m_targetVector);
   ZeroMemory(classes);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CRegressorNets::RegressorNetsFF(vector &V_in)
  {
   matrix L_INPUT = {}, L_OUTPUT= {}, L_WEIGHTS = {};
   vector v_weights = {};

   ulong start = 0;

   L_INPUT = matrix_utils.VectorToMatrix(V_in);

   vector L_BIAS_VECTOR = {};
   matrix L_BIAS_MATRIX = {};

   ulong b_start = 0;

   for(ulong i=0; i<W_CONFIG.Size(); i++)
     {
      matrix_utils.Copy(W,v_weights,start,ulong(W_CONFIG[i]));

      L_WEIGHTS = matrix_utils.VectorToMatrix(v_weights,L_INPUT.Rows());

      matrix_utils.Copy(B,L_BIAS_VECTOR,b_start,(ulong)HL_CONFIG[i]);
      L_BIAS_MATRIX = matrix_utils.VectorToMatrix(L_BIAS_VECTOR);

      #ifdef DEBUG_MODE
         Print("--? ",i);
         Print("L_WEIGHTS\n",L_WEIGHTS,"\nL_INPUT\n",L_INPUT,"\nL_BIAS\n",L_BIAS_MATRIX);
      #endif 
     
      L_OUTPUT = L_WEIGHTS.MatMul(L_INPUT); //Inputs x Weights

//--- 

      L_OUTPUT = L_OUTPUT+L_BIAS_MATRIX; //Add bias
      L_OUTPUT.Activation(L_OUTPUT, ENUM_ACTIVATION_FUNCTION(A_FX));
      
//---

      L_INPUT.Copy(L_OUTPUT); //Assign outputs to the inputs
      start += (ulong)W_CONFIG[i]; //New weights copy
      b_start += (ulong)HL_CONFIG[i];

     }
     
   #ifdef DEBUG_MODE 
      Print("outputs\n ",L_OUTPUT);
   #endif 
   
   return(L_OUTPUT[0,0]);
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
