//+------------------------------------------------------------------+
//|                                                 selftrain NN.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\preprocessing.mqh>;
#include <MALE5\matrix_utils.mqh>;
#include <MALE5\metrics.mqh>

#define  RANDOM_STATE 42

//+------------------------------------------------------------------+
//| defines                                                          |
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
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CRegNeuralNets
  {
   CMatrixutils   matrix_utils;
   CPreprocessing *normalize_x;
   CPreprocessing *normalize_y;
   CMetrics  metrics;
   
   private:
      matrix            W;
      matrix            B;
      uint              m_inputs;
      ulong             m_rows;
      bool              data_norm;
      bool              isBackProp;
      matrix            m_x_matrix;
      vector            m_y_vector;
      
      loss  L_FX;
      activation  A_FX;
      norm_technique    scaler;         
      
   public:
                        CRegNeuralNets(matrix &xmatrix, vector &yvector, double alpha, uint epochs, activation ACTIVATION_FUNCTION, loss LOSS_FUNCTIONm, norm_technique NORM_METHOD);
                       ~CRegNeuralNets(void);
                        
                        matrix ForwardPass(vector &input_v);
                        vector ForwardPass(matrix &matrix_);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRegNeuralNets::CRegNeuralNets(matrix &xmatrix, vector &yvector,double alpha, uint epochs, activation ACTIVATION_FUNCTION, loss LOSS_FUNCTION, norm_technique NORM_METHOD)
 {
   A_FX = ACTIVATION_FUNCTION;
   L_FX = LOSS_FUNCTION;
   
   m_x_matrix = xmatrix;
   m_y_vector = yvector;
   
//--- Weights Initialization
   
   m_inputs = (uint)m_x_matrix.Cols();
   m_rows = m_x_matrix.Rows();
   
   this.W = matrix_utils.Random(0.0, 1.0,1,m_inputs, RANDOM_STATE);
   //this.W = this.W * sqrt(2/((double)m_inputs + 1)); //glorot
   
   this.W = this.W * 1/sqrt(m_inputs); //He initialization
   
   this.B = matrix_utils.Random(0.0, 0.5,1,1,RANDOM_STATE);
   
   matrix YMatrix = matrix_utils.VectorToMatrix(m_y_vector);
   
    if (NORM_METHOD != NORM_NONE)
        {
          data_norm = true;
          normalize_x = new CPreprocessing(m_x_matrix, NORM_METHOD); 
          YMatrix = matrix_utils.VectorToMatrix(m_y_vector);
          normalize_y = new CPreprocessing(YMatrix,NORM_METHOD);
          
          m_y_vector = matrix_utils.MatrixToVector(YMatrix);
        }
    
   
//--- Training the NN
   
   matrix DX_W={}, LOSS_DX(1,1);  
   matrix INPUT;
   
   vector pred(1), actual(1);
   vector preds(m_rows), actuals(m_rows);
   
   matrix OUTPUT = {};
   
   isBackProp = true;
   for (ulong epoch=0; epoch<epochs && !IsStopped(); epoch++)
      {
         for (ulong iter=0; iter<m_rows; iter++)
            {
              OUTPUT = ForwardPass(m_x_matrix.Row(iter));
              pred = matrix_utils.MatrixToVector(OUTPUT);
              
              actual[0] = m_y_vector[iter];
              
              preds[iter] = pred[0];
              actuals[iter] = actual[0];
              
           //---
              
              INPUT = matrix_utils.VectorToMatrix(m_x_matrix.Row(iter));
              
              vector loss_v = pred.LossGradient(actual, ENUM_LOSS_FUNCTION(L_FX));
              
              LOSS_DX.Col(loss_v, 0);
              
              OUTPUT.Derivative(OUTPUT, ENUM_ACTIVATION_FUNCTION(A_FX));
              
              //printf("loss[%dx%d] x Derivatives [%dx%d] ",LOSS_DX.Rows(),LOSS_DX.Cols(), OUTPUT.Rows(), OUTPUT.Cols());
              
              OUTPUT = LOSS_DX * OUTPUT;
              
              INPUT = INPUT.Transpose();
              DX_W = OUTPUT.MatMul(INPUT);
               
              this.W -= (alpha * DX_W);
            }
         
         
         printf("[ %d/%d ] Loss = %.8f | accuracy %.3f ",epoch+1,epochs,preds.Loss(actuals,ENUM_LOSS_FUNCTION(L_FX)),metrics.r_squared(actuals, preds));
      }
   isBackProp = false; //Training has fininshed
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRegNeuralNets::~CRegNeuralNets(void)
 {
   data_norm = false;
   isBackProp = false;
   
   delete (normalize_x);
   delete (normalize_y);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CRegNeuralNets::ForwardPass(vector &input_v)
 {
   if (!isBackProp && data_norm) 
      normalize_x.Normalization(input_v);
   
   matrix INPUT = this.matrix_utils.VectorToMatrix(input_v);
   matrix OUTPUT;
   
   //Print("INPUT\n",INPUT,"\nW\n",W,"\nB\n",B);
   
   OUTPUT = W.MatMul(INPUT);
   
   OUTPUT = OUTPUT + B;
   
   OUTPUT.Activation(OUTPUT, ENUM_ACTIVATION_FUNCTION(A_FX));
   
   if (!isBackProp && data_norm)
    {
      //Print("Inside Normalize data ");
      vector output_v = matrix_utils.MatrixToVector(OUTPUT);  
      normalize_y.ReverseNormalization(output_v);
      
      OUTPUT = matrix_utils.VectorToMatrix(output_v);
    }
   
   //Print("OUTPUT\n",OUTPUT);
   
   return (OUTPUT);
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CRegNeuralNets::ForwardPass(matrix &matrix_)
 {
   ulong size = matrix_.Rows();
   
   vector v(size);
   matrix out={};
   
    for (ulong i=0; i<size; i++)
       {
         out = ForwardPass(matrix_.Row(i));
         v[i] = out[0][0]; 
       }
    return (v);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

