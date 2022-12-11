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

#include <preprocessing.mqh>;
#include <matrix_utils.mqh>;

CPreprocessing pre_processing;
CMatrixutils   matrix_utils;

//+------------------------------------------------------------------+
 

enum LOSS_TYPE
  {
    MSE, //for regression
    BCE  //for classification
  } L_FX;

//---

enum nn_optimizers
  {
     ADAM,
     SGD
  } nn_optimizer;

//+------------------------------------------------------------------+

class CNeuralNets
  {
   protected:
                  matrix                   XMatrix;
                  matrix                   YMatrix;
                  
                  ulong                    m_hlnodes;
                  vector                   m_targetVector;
                  
                  ulong                    m_rows, m_cols;
                  uint                     m_outputs;  //MLP outputs
                  matrix<double>           w1; //Weights on the input layer to hidden
                  matrix<double>           w2; //weights on the hidden layer to output
                  
                  matrix HL_MATRIX; //hidden layer matrix
                  matrix OL_MATRIX; //Output layer matrix
                  
                  ENUM_ACTIVATION_FUNCTION A_FX; 

                  matrix OneHotEncoding(vector &v, uint &classes);
                  uint   OutputNodeDxd(ENUM_ACTIVATION_FUNCTION ActivationFx,uint dt_clas);
                  
   public:
                  CNeuralNets(matrix &Matrix, ENUM_ACTIVATION_FUNCTION ActivationFX, ulong hl_nodes, LOSS_TYPE loss_function);
                 ~CNeuralNets(void);
                  
                  
                  vector FeedForwardMLP(vector& V_in);       
                  void   TrainFeedForwardMLP(matrix& mat_out);
                  void   BackPropagation(nn_optimizers optimizer, double alpha=0.01, uint iterations = 100);
                           
  };

//+------------------------------------------------------------------+

CNeuralNets::CNeuralNets(matrix &Matrix, ENUM_ACTIVATION_FUNCTION ActivationFX, ulong hl_nodes, LOSS_TYPE loss_function)
 {
    A_FX = ActivationFX;
    L_FX = loss_function;
    m_hlnodes = hl_nodes;
    
    m_rows = Matrix.Rows();
    m_cols = Matrix.Cols(); 

//---
  
    XMatrix.Copy(Matrix);
    XMatrix.Reshape(m_rows,m_cols-1); //remove target variable
    pre_processing.MinMaxScaler(XMatrix);
    
    vector y = Matrix.Col(m_cols-1);
    YMatrix = OneHotEncoding(y,m_outputs);
  
//---
   
    //uint dt_classes = PrepareData(Matrix,YMatrix);
    //m_outputs = OutputNodeDxd(ActivationFX,dt_classes); 
    //the output nodes on the final layer of a MLP depends on the classes in dataset and 
    //the king of activation function selected for the network

//---
    
    
   #ifdef  DEBUG_MODE
     {
       m_cols = m_cols-1;
       w1.Resize(m_hlnodes,m_cols);
       
       ulong count1 = 1;
       
       for (ulong i=0; i<w1.Rows(); i++)
         for (ulong j=0; j<w1.Cols(); j++, count1++)
             {
               w1[i][j] = (double)count1/10.0;
             }
       
       w2.Resize(m_outputs,m_hlnodes);
       
       for (ulong i=0, count=count1; i<w2.Rows(); i++)
         for (ulong j=0; j<w2.Cols(); j++,count++)
            { 
               w2[i][j] = (double)count/10.0;
            }
     } 
   #endif
    
//---

   #ifdef  DEBUG_MODE
   
         Print("w1 matrix\n",w1,"\nw2 matrix\n",w2);
     
         printf("Init A_FX %s hl_nodes %d Loss_FX %s Output Nodes %d",EnumToString(ActivationFX),hl_nodes,EnumToString(loss_function),m_outputs);
         Print("xmatrix\n",XMatrix,"\nYmatrix\n",YMatrix); 
         
   #endif      

 }

//+------------------------------------------------------------------+

CNeuralNets::~CNeuralNets(void)
 {
   ZeroMemory(XMatrix);
   ZeroMemory(YMatrix);
 }


//+------------------------------------------------------------------+

uint CNeuralNets::OutputNodeDxd(ENUM_ACTIVATION_FUNCTION ActivationFx,uint dt_clas)
 {
    switch(ActivationFx)
      {
       case  AF_ELU:           return 1;          break;  //Exponential Linear Unit
       case  AF_EXP:           return 1;          break;  //Exponential
       case  AF_GELU:          return 1;          break;  //Gaussian Error Linear Unit
       case AF_LINEAR:         return 1;          break;  //Linear
       case AF_LRELU:          return 1;          break;  //Leaky Rectified linear unit
       case AF_RELU:           return 1;          break;  //Rectified linear unit
       case AF_SELU:           return 1;          break;  //Scaled exponential linear unit
       case AF_TRELU:          return 1;          break;  //Threshold Rectified linear unit
       case AF_SOFTPLUS:       return 1;          break;  //Softplus
       
       case AF_HARD_SIGMOID:  return dt_clas;          break;  //Hard Sigmoid 
       case AF_SIGMOID:       return dt_clas;          break;  //Sigmoid
       case AF_SWISH:         return dt_clas;          break;  //Swish
       case AF_SOFTSIGN:      return dt_clas;          break;  //Softsign
       case AF_TANH:          return dt_clas;          break;  //The hyperbolic tangent Function
       
       default:
         Print("Unknown Activation Function");
         break;
      }
  return(0); 
 }

//+------------------------------------------------------------------+

vector CNeuralNets::FeedForwardMLP(vector &V_in)
 {    
   V_in.Resize(V_in.Size()); //This line should be removed
   
   matrix m_in = matrix_utils.VectorToMatrix(V_in); //Input matrix
   //Print("w1\n",w1,"\nm_in\n",m_in);
   
   HL_MATRIX = w1.MatMul(m_in);
   HL_MATRIX.Activation(HL_MATRIX,A_FX);  //Passing them to Activation F(x)
      
   //Print("w2\n",w2,"\nHL_MATRIX\n",HL_MATRIX);
      
   OL_MATRIX = w2.MatMul(HL_MATRIX);
   OL_MATRIX.Activation(OL_MATRIX,A_FX); //passing the output to an Activation F(x)
   
   return(matrix_utils.MatrixToVector(OL_MATRIX));
 }
 
//+------------------------------------------------------------------+

void CNeuralNets::BackPropagation(nn_optimizers optimizer, double alpha=0.010000,uint iterations=100)
 {   
   double cost=0;
   
   nn_optimizer = optimizer;
   
   XMatrix.Resize(XMatrix.Rows(),2);
   
    for (uint iters=0; iters<iterations; iters++)
      {
        vector v_in, v_out={}, y_v = {}; 
        matrix DXOUT_MATRIX, DXHL_MATRIX, DX_DY;
        matrix w2_temp = w2;
        matrix INPUT_MATRIX;
        
        //Print("w1\n",w1,"\nw2\n",w2);
        
        for(ulong i=0; i<m_rows; i++) //Loop the entire dataset
           {
              v_in = XMatrix.Row(i);
              y_v = YMatrix.Row(i);
              
              INPUT_MATRIX = matrix_utils.VectorToMatrix(v_in);
              
              v_out = FeedForwardMLP(v_in);
              
              HL_MATRIX.Derivative(DXHL_MATRIX,A_FX);
              
              //dx_out = v_out - y_v;
              DXOUT_MATRIX = matrix_utils.VectorToMatrix(v_out - y_v);
              
              //Print("DXOUT MATRIX\n",DXOUT_MATRIX,"\nDXHL\n",DXHL_MATRIX);
              
              DXHL_MATRIX = DXHL_MATRIX.Transpose();
              
              DX_DY = DXOUT_MATRIX.MatMul(DXHL_MATRIX);// * HL_MATRIX;
              DX_DY = DX_DY.MatMul(HL_MATRIX);

              //Print("DX_DY\n",DX_DY);
              
              //w2 = w2 - (DX_DY * alpha);
              for (ulong r=0; r<w2.Rows(); r++)
                 for (ulong c=0; c<w2.Cols(); c++)
                   {
                     w2[r][c] = w2[r][c] - DX_DY[r][0] * alpha;
                   }

//--- w2

              DXOUT_MATRIX = DXOUT_MATRIX.Transpose();
              //Print("DXOUT_MATRIX\n",DXOUT_MATRIX,"\nw2_temp\n",w2_temp);
              DX_DY = DXOUT_MATRIX.MatMul(w2_temp);
              
              DXHL_MATRIX = DXHL_MATRIX.Transpose();
              //Print("DX_DY\n",DX_DY,"\nDXHL_MATRIX\n",DXHL_MATRIX);
              
              DX_DY = DX_DY.MatMul(DXHL_MATRIX);
              
              //INPUT_MATRIX = INPUT_MATRIX.Transpose();
              
              //Print("dxdy\n",DX_DY,"\nINPUT_MATRIX\n",INPUT_MATRIX);
              
              DX_DY = INPUT_MATRIX.MatMul(DX_DY);
              
              //Print("DX_DY\n",DX_DY,"\nw1\n",w1);     
              
              for (ulong r=0; r<w1.Rows(); r++)
                 for (ulong c=0; c<w1.Cols(); c++)
                     w1[r][c] = w1[r][c] - DX_DY[c][0] * alpha;
              
           }
        
            #ifdef  DEBUG_MODE
                       Print("------> Loss = ",v_out.Loss(y_v,LOSS_BCE));
            #endif 
      }
 }
 
//+------------------------------------------------------------------+ 

matrix CNeuralNets::OneHotEncoding(vector &v, uint &classes)
 {
   matrix mat = {}; 
   
//---

   vector temp_t = v, v_classes = {v[0]};

   for(ulong i=0, count =1; i<v.Size(); i++)  //counting the different neighbors
     {
      for(ulong j=0; j<v.Size(); j++)
        {
         if(v[i] == temp_t[j] && temp_t[j] != -1000)
           {
            bool count_ready = false;

            for(ulong n=0; n<v_classes.Size(); n++)
               if(v[i] == v_classes[n])
                    count_ready = true;

            if(!count_ready)
              {
               count++;
               v_classes.Resize(count);

               v_classes[count-1] = v[i];
               //Print("v ",v[i]);

               temp_t[j] = -1000; //modify so that it can no more be counted
              }
            else
               break;
            //Print("t vectors vector ",v);
           }
         else
            continue;
        }
     }
     
//---

     classes = (uint)v_classes.Size();
     mat.Resize(v.Size(),v_classes.Size());
     mat.Fill(-100);
     
     for (ulong i=0; i<mat.Rows(); i++)
        for (ulong j=0; j<mat.Cols(); j++)
           {
               if (v[i] == v_classes[j])
                  mat[i][j] = 1;
               else 
                  mat[i][j] = 0;     
           }
   
   return(mat);
 }
 
//+------------------------------------------------------------------+

 