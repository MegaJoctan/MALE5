//+------------------------------------------------------------------+
//|                                                          svm.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\preprocessing.mqh>
#include <MALE5\MatrixExtend.mqh>
#include <MALE5\Metrics.mqh>
//#include <MALE5\kernels.mqh>

//+------------------------------------------------------------------+
//|  At its core, SVM aims to find a hyperplane that best separates  |
//|  two classes of data points in a high-dimensional space.         |
//|  This hyperplane is chosen to maximize the margin between the    |
//|  two classes, making it the optimal decision boundary.           |
//+------------------------------------------------------------------+

#define UNDEFINED_REPLACE 1


class CLinearSVM
  {
   protected:
         
      vector            W;
      double            B; 
      
      bool is_fitted_already;
      bool during_training;
      
      struct svm_config 
        {
          uint batch_size;
          double alpha;
          double lambda;
          uint epochs;
        };

   private:
      svm_config config;
   
   protected:
        
      
                        double hyperplane(vector &x);
                        
   public:
                        CLinearSVM(uint batch_size=32, double alpha=0.001, uint epochs= 1000,double lambda=0.1);
                       ~CLinearSVM(void);
                        
                        void fit(matrix &x, vector &y);
                        int Predict(vector &x);
                        vector Predict(matrix &x);
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

CLinearSVM::CLinearSVM(uint batch_size=32, double alpha=0.001, uint epochs= 1000,double lambda=0.1)
 {   
    is_fitted_already = false;
    during_training = false;
    
    config.batch_size = batch_size;
    config.alpha = alpha;
    config.lambda = lambda;
    config.epochs = epochs;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

CLinearSVM::~CLinearSVM(void)
 {
 
 }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLinearSVM::hyperplane(vector &x)
 {
   return x.MatMul(W) - B;   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CLinearSVM::Predict(vector &x)
 { 
   if (!is_fitted_already)
     {
       Print("Err | The model is not trained, call the fit method to train the model before you can use it");
       return 1000;
     }
     
   return MatrixExtend::Sign(hyperplane(x));
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLinearSVM::Predict(matrix &x)
 {
   vector v(x.Rows());
   
   for (ulong i=0; i<x.Rows(); i++)
     v[i] = Predict(x.Row(i));
     
   return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLinearSVM::fit(matrix &x, vector &y)
 {  
   ulong rows = x.Rows(),
         cols = x.Cols();
   
   if (x.Rows() != y.Size())
      {
         Print("Support vector machine Failed | FATAL | x m_rows not same as yvector size");
         return;
      }
   
   W.Resize(cols);
   B = 0;
     
//---

  if (rows < config.batch_size)
    {
      Print("The number of samples/rows in the dataset should be less than the batch size");
      return;
    }
   
    matrix temp_x;
    vector temp_y;
    matrix w, b;
    
    vector preds = {};
    vector loss(config.epochs);
    during_training = true;

    for (uint epoch=0; epoch<config.epochs; epoch++)
      {
        
         for (uint batch=0; batch<=(uint)MathFloor(rows/config.batch_size); batch+=config.batch_size)
           {              
              temp_x = MatrixExtend::Get(x, batch, (config.batch_size+batch)-1);
              temp_y = MatrixExtend::Get(y, batch, (config.batch_size+batch)-1);
              
              #ifdef DEBUG_MODE:
                  Print("x\n",temp_x,"\ny\n",temp_y);
              #endif 
              
               for (uint sample=0; sample<temp_x.Rows(); sample++)
                  {                                        
                     // yixiw-b≥1
                     
                      if (temp_y[sample] * hyperplane(temp_x.Row(sample))  >= 1) 
                        {
                          this.W -= config.alpha * (2 * config.lambda * this.W); // w = w + α* (2λw - yixi)
                        }
                      else
                         {
                           this.W -= config.alpha * (2 * config.lambda * this.W - ( temp_x.Row(sample) * temp_y[sample] )); // w = w + α* (2λw - yixi)
                           
                           this.B -= config.alpha * temp_y[sample]; // b = b - α* (yi)
                         }  
                  }
           }
        
        //--- Print the loss at the end of an epoch
       
         is_fitted_already = true;  
         
         preds = this.Predict(x);
         
         loss[epoch] = preds.Loss(y, LOSS_BCE);
        
         printf("---> epoch [%d/%d] Loss = %f Accuracy = %f",epoch+1,config.epochs,loss[epoch],Metrics::accuracy_score(y, preds));
         
        #ifdef DEBUG_MODE:  
          Print("W\n",W," B = ",B);  
        #endif   
      }
    
    during_training = false;
    
    return;
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|             SVM DUAL | for non linear problems  made with ONNX   |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
/*
class CDualSVMONNX
  {
private:
      CPreprocessing<vectorf, matrixf> *normalize_x;
      CMatrixutils matrix_utils;
      
      struct data_struct
       {
         ulong rows,
               cols;
       } df;
      
public:  
                     CDualSVMONNX(void);
                    ~CDualSVMONNX(void);
      
                     long onnx_handle;              
                     
                     void SendDataToONNX(matrixf &data, string csv_name = "DualSVMONNX-data.csv", string csv_header="");
                     bool LoadONNX(const uchar &onnx_buff[], ENUM_ONNX_FLAGS flags, vectorf &norm_max, vectorf &norm_min);
                     int Predict(vectorf &inputs);
                     vector Predict(matrixf &inputs);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDualSVMONNX::CDualSVMONNX(void)
 {
   
 }
//+------------------------------------------------------------------+
//|   This function better be called inside a script responsible for |
//| preparing data for training models in python side of things      |
//+------------------------------------------------------------------+
void CDualSVMONNX::SendDataToONNX(matrixf &data, string csv_name = "DualSVMONNX-data.csv", string csv_header="")
 {
    df.cols = data.Cols();
    df.rows = data.Rows();
    
    if (df.cols == 0 || df.rows == 0)
      {
         Print(__FUNCTION__," data matrix invalid size ");
         return;
      }
    
    matrixf split_x;
    vectorf  split_y;
    
    MatrixExtend::XandYSplitMatrices(data, split_x, split_y); //since we are going to be normalizing the independent variable only we need to split the data into two
    
    normalize_x = new CPreprocessing<vectorf,matrixf>(split_x, NORM_MIN_MAX_SCALER); //Normalizing Independent variable only
    
    
    matrixf new_data = split_x;
    new_data.Resize(data.Rows(), data.Cols());
    new_data.Col(split_y, data.Cols()-1);
    
    if (csv_header == "")
      {
         for (ulong i=0; i<df.cols; i++)
           csv_header += "COLUMN "+string(i+1) + (i==df.cols-1 ? "" : ","); //do not put delimiter on the last column
      }
    
//--- Save the Normalization parameters also
    
   matrixf params = {};
    
   string sep=",";
   ushort u_sep;
   string result[];
   
   u_sep=StringGetCharacter(sep,0); 
   int k=StringSplit(csv_header,u_sep,result); 
   
   ArrayRemove(result, k-1, 1); //remove the last column header since we do not have normalization parameters for the target variable  as it is not normalized
    
    normalize_x.min_max_scaler.min.Swap(params);
    MatrixExtend::WriteCsv("min_max_scaler.min.csv",params,result,false,8);
    normalize_x.min_max_scaler.max.Swap(params);
    MatrixExtend::WriteCsv("min_max_scaler.max.csv",params,result,false,8); 
    
//--- 
    
    MatrixExtend::WriteCsv(csv_name, new_data, csv_header, false, 8); //Save dataset to a csv file 
 }
//+------------------------------------------------------------------+
//|  Loads Onnx model from  a resource uchar Array | This function   |
//| need to be loaded inside OnInit function of an EA or indicator   |
//+------------------------------------------------------------------+
bool CDualSVMONNX::LoadONNX(const uchar &onnx_buff[], ENUM_ONNX_FLAGS flags, vectorf &norm_max, vectorf &norm_min)
 {
   onnx_handle =  OnnxCreateFromBuffer(onnx_buff, flags); //creating onnx handle buffer 
   
   if (onnx_handle == INVALID_HANDLE)
    {
       Print(__FUNCTION__," OnnxCreateFromBuffer Error = ",GetLastError());
       return false;
    }
   
//---
   
   const long inputs[] = {1,4};
   
   if (!OnnxSetInputShape(onnx_handle, 0, inputs)) //Giving the Onnx handle the input shape
     {
       Print(__FUNCTION__," Failed to set the input shape Err=",GetLastError());
       return false;
     }
   
   long outputs_0[] = {1};
   if (!OnnxSetOutputShape(onnx_handle, 0, outputs_0)) //giving the onnx handle first node output shape
     {
       Print(__FUNCTION__," Failed to set the output shape Err=",GetLastError());
       return false;
     }
     
   long outputs_1[] = {1,2};
   if (!OnnxSetOutputShape(onnx_handle, 1, outputs_1)) //giving the onnx handle second node output shape
     {
       Print(__FUNCTION__," Failed to set the output shape Err=",GetLastError());
       return false;
     }
   
   normalize_x = new CPreprocessing<vectorf,matrixf>(norm_max, norm_min); //Load min max scaler with parameters
    
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDualSVMONNX::~CDualSVMONNX(void)
 {
   delete (normalize_x);
   
   if (onnx_handle != 0)
      OnnxRelease(onnx_handle);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CDualSVMONNX::Predict(vectorf &inputs)
 {
    vectorf outputs(1); //label outputs
    vectorf x_output(2); //probabilities
    
    vectorf temp_inputs = inputs;
    
    normalize_x.Normalization(temp_inputs); //Normalize the input features
    
    if (!OnnxRun(onnx_handle, ONNX_DEFAULT, temp_inputs, outputs, x_output))
      {
         Print("Failed to get predictions from onnx Err=",GetLastError());
         return (int)outputs[0];
      }
      
   return (int)outputs[0];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDualSVMONNX::Predict(matrixf &inputs)
 {
   vector vec(inputs.Rows());
    for (ulong i=0; i<inputs.Rows(); i++)
      vec[i] = Predict(inputs.Row(i));
      
   return vec;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+






//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|             SVM DUAL | for non linear problems                   |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+
/*
class CDualSVM: protected CLinearSVM
  {
private:
   
   __kernels__       *kernel;
   
   struct dual_svm_config: svm_config //Inherit configs from Linear SVM
    {  
       kernels kernel;
       uint degree;
       double sigma;
       double beta;
    };
   
   dual_svm_config config;
   
   matrix x;
   vector y;
   
   vector y_labels;
   vector model_alpha;
   
   int decision_function(vector &x);

   matrix VectorToMatrix(vector &v)
    {
      matrix ret_m;
      vector temp_v = v;
      
      temp_v.Swap(ret_m);
      
      return ret_m;
    }
      
    double MatrixToDBL(matrix &mat)
    {   
      if (mat.Rows()>1 || mat.Cols()>1)
       {
         Print(__FUNCTION__," Can't convert matrix to double as this is not a 1x1 matrix");
         return 0;
       }
      return mat[0][0];
    }
          
public:
                    CDualSVM(kernels KERNEL, double alpha, double beta, uint degree, double sigma, uint batch_size=32, uint epochs= 1000);
                   ~CDualSVM(void);
                    
                    void fit(matrix &x, vector &y);
                    vector Predict(matrix &x);
                    int Predict(vector &x);

  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDualSVM::CDualSVM(kernels KERNEL,
                   double alpha, 
                   double beta, 
                   uint  degree, 
                   double sigma,
                   uint batch_size=32, 
                   uint epochs= 1000
                   )
 {
    kernel = new __kernels__(KERNEL, alpha, beta, degree, sigma);
   
    config.kernel = KERNEL;
    config.alpha = alpha; 
    config.beta = beta;
    config.degree = degree; 
    config.sigma = sigma;
    config.batch_size = batch_size;
    config.epochs = epochs;
    
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDualSVM::~CDualSVM(void)
 {
   delete (kernel);
   delete (normalize_x);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CDualSVM::decision_function(vector &x)
 { 
   vector tempx =x;
   matrix x_;
   tempx.Swap(x_);
   
   //Print("x\n",this.x," x_ ",x_);
   
   matrix kernel_res = this.kernel.KernelFunction(this.x, x_);
   
   //printf("alpha (%dx%d) y_label (%dx%d) kernel_res =(%dx%d)",VectorToMatrix(model_alpha).Rows(),VectorToMatrix(model_alpha).Cols(), VectorToMatrix(y_labels).Rows(), VectorToMatrix(y_labels).Cols(),kernel_res.Rows(),kernel_res.Cols());
   
   return sign(MatrixToDBL(VectorToMatrix(model_alpha * this.y).MatMul(kernel_res)));
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

int CDualSVM::Predict(vector &x)
 { 
   if (!is_fitted_already)
     {
       Print("Err | The model is not trained, call the fit method to train the model before you can use it");
       return 1000;
     }
   
   if (x.Size() <=0)
     {
       Print(__FUNCTION__," Err invalid x size ");
       return 1e3;
     }
   return decision_function(x);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDualSVM::Predict(matrix &x)
 {
   vector v(x.Rows());
   
   for (ulong i=0; i<x.Rows(); i++)
     v[i] = Predict(x.Row(i));
     
   return v;
 }
//+------------------------------------------------------------4------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDualSVM::fit(matrix &x,vector &y)
 {
   x = x;
   y = y;
   

   y_labels = this.MatrixExtend::Classes(y);
   
   
   ulong rows = x.Rows(), 
         cols = x.Cols();
         
   model_alpha = MatrixExtend::Zeros(rows);
   
   
   if (x.Rows() != y.Size())
      {
         Print("Support vector machine Failed | FATAL | x m_rows not same as yvector size");
         return;
      }
   
   W.Resize(cols);
   B = 0;
   
   normalize_x = new CPreprocessing<vector, matrix>(x, NORM_STANDARDIZATION);
     
//---

  
  if (rows < config.batch_size)
    {
      Print("The number of samples/rows in the dataset should be less than the batch size");
      return;
    }
   
    matrix temp_x;
    vector temp_y;
    matrix w, b;
    
    vector preds = {};
    vector loss(config.epochs);
    vector ones(rows);
    ones.Fill(1); 
    
    for (uint epoch=0; epoch<config.epochs; epoch++)
      {
        vector gradient = {};
        
         for (uint batch=0; batch<=(uint)MathFloor(rows/config.batch_size); batch+=config.batch_size)
           {
            
              temp_x = MatrixExtend::Get(x, batch, (config.batch_size+batch)-1);
              temp_y = MatrixExtend::Get(y, batch, (config.batch_size+batch)-1);
              
              #ifdef DEBUG_MODE:
                  Print("x\n",temp_x,"\ny\n",temp_y);
              #endif 
              
              for (uint sample=0; sample<temp_x.Rows(); sample++)
                {
                   //printf("outer alpha =(%dx%d) y_outer =(%dx%d)",model_alpha.Outer(model_alpha).Rows(),model_alpha.Outer(model_alpha).Cols(),y.Outer(y).Rows(),y.Outer(y).Cols());
                   
                   gradient = ones - (model_alpha.Outer(model_alpha) * y.Outer(y) * this.kernel.KernelFunction(x, x)).Sum();
                          
                   model_alpha += config.alpha * gradient;
                   model_alpha.Clip(0, INT_MAX);
               }
           }
           
        //--- Print the loss at the end of an epoch
       
         is_fitted_already = true;  
         
         //preds = this.Predict(x);
         
         loss[epoch] = preds.Loss(this.y, LOSS_BCE);
        
         printf("---> epoch [%d/%d] Loss = %f ",epoch+1,config.epochs,loss[epoch]);
         
        #ifdef DEBUG_MODE:  
          Print("W\n",W," B = ",B);  
        #endif   
        
      }
    
   Print("Optimal Lagrange Multipliers (alpha):", model_alpha);
      
    return;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
*/
