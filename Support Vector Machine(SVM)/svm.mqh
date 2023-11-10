//+------------------------------------------------------------------+
//|                                                          svm.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\preprocessing.mqh>
#include <MALE5\matrix_utils.mqh>
#include <MALE5\metrics.mqh>
#include <MALE5\kernels.mqh>

//+------------------------------------------------------------------+
//|  At its core, SVM aims to find a hyperplane that best separates  |
//|  two classes of data points in a high-dimensional space.         |
//|  This hyperplane is chosen to maximize the margin between the    |
//|  two classes, making it the optimal decision boundary.           |
//+------------------------------------------------------------------+

#define RANDOM_STATE 42

class CLinearSVM
  {
   protected:
   
      CMatrixutils      matrix_utils;
      CMetrics          metrics;
      
      CPreprocessing<vector, matrix> *normalize_x;
      
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
                        
                        int sign(double var);
                        vector sign(const vector &vec);
                        matrix sign(const matrix &mat);
                        
   public:
                        CLinearSVM(uint batch_size=32, double alpha=0.001, uint epochs= 1000,double lambda=0.1);
                       ~CLinearSVM(void);
                        
                        void fit(matrix &x, vector &y);
                        int predict(vector &x);
                        vector predict(matrix &x);
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
   delete (normalize_x);
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
int CLinearSVM::predict(vector &x)
 { 
   if (!is_fitted_already)
     {
       Print("Err | The model is not trained, call the fit method to train the model before you can use it");
       return 1000;
     }
   
   vector temp_x = x;
   if (!during_training)
     normalize_x.Normalization(temp_x);
     
   return sign(hyperplane(temp_x));
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLinearSVM::predict(matrix &x)
 {
   vector v(x.Rows());
   
   for (ulong i=0; i<x.Rows(); i++)
     v[i] = predict(x.Row(i));
     
   return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLinearSVM::fit(matrix &x, vector &y)
 {
   matrix X = x;
   vector Y = y;
  
   ulong rows = X.Rows(),
         cols = X.Cols();
   
   if (X.Rows() != Y.Size())
      {
         Print("Support vector machine Failed | FATAL | X m_rows not same as yvector size");
         return;
      }
   
   W.Resize(cols);
   B = 0;
    
   normalize_x = new CPreprocessing<vector, matrix>(X, NORM_STANDARDIZATION); //Normalizing independent variables
     
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
              temp_x = matrix_utils.Get(X, batch, (config.batch_size+batch)-1);
              temp_y = matrix_utils.Get(Y, batch, (config.batch_size+batch)-1);
              
              #ifdef DEBUG_MODE:
                  Print("X\n",temp_x,"\ny\n",temp_y);
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
         
         preds = this.predict(X);
         
         loss[epoch] = preds.Loss(Y, LOSS_BCE);
        
         printf("---> epoch [%d/%d] Loss = %f Accuracy = %f",epoch+1,config.epochs,loss[epoch],metrics.confusion_matrix(Y, preds, false));
         
        #ifdef DEBUG_MODE:  
          Print("W\n",W," B = ",B);  
        #endif   
      }
    
    during_training = false;
    
    return;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CLinearSVM::sign(double var)
 {
   //Print("Sign input var = ",var);
   
   if (var == 0)
    return (0);
   else if (var < 0)
    return -1;
   else 
    return 1; 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLinearSVM::sign(const vector &vec)
 {
   vector ret = vec;
   
   for (ulong i=0; i<vec.Size(); i++)
     ret[i] = sign((int)vec[i]);
   
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CLinearSVM::sign(const matrix &mat)
 { 
   matrix ret = mat;
   
   for (ulong i=0; i<mat.Rows(); i++)
     for (ulong j=0; j<mat.Cols(); j++)
        ret[i][j] = sign((int)mat[i][j]); 
        
   return ret;
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|             SVM DUAL | for non linear problems                   |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

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
   
   matrix X;
   vector Y;
   
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
                    vector predict(matrix &x);
                    int predict(vector &x);

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
   
   //Print("x\n",this.X," x_ ",x_);
   
   matrix kernel_res = this.kernel.KernelFunction(this.X, x_);
   
   //printf("alpha (%dx%d) y_label (%dx%d) kernel_res =(%dx%d)",VectorToMatrix(model_alpha).Rows(),VectorToMatrix(model_alpha).Cols(), VectorToMatrix(y_labels).Rows(), VectorToMatrix(y_labels).Cols(),kernel_res.Rows(),kernel_res.Cols());
   
   return sign(MatrixToDBL(VectorToMatrix(model_alpha * this.Y).MatMul(kernel_res)));
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

int CDualSVM::predict(vector &x)
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
vector CDualSVM::predict(matrix &x)
 {
   vector v(x.Rows());
   
   for (ulong i=0; i<x.Rows(); i++)
     v[i] = predict(x.Row(i));
     
   return v;
 }
//+------------------------------------------------------------4------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDualSVM::fit(matrix &x,vector &y)
 {
   X = x;
   Y = y;
   

   y_labels = this.matrix_utils.Classes(Y);
   
   
   ulong rows = X.Rows(), 
         cols = X.Cols();
         
   model_alpha = matrix_utils.Zeros(rows);
   
   
   if (X.Rows() != Y.Size())
      {
         Print("Support vector machine Failed | FATAL | X m_rows not same as yvector size");
         return;
      }
   
   W.Resize(cols);
   B = 0;
   
   normalize_x = new CPreprocessing<vector, matrix>(X, NORM_STANDARDIZATION);
     
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
            /*
              temp_x = matrix_utils.Get(X, batch, (config.batch_size+batch)-1);
              temp_y = matrix_utils.Get(Y, batch, (config.batch_size+batch)-1);
              
              #ifdef DEBUG_MODE:
                  Print("X\n",temp_x,"\ny\n",temp_y);
              #endif 
              
              for (uint sample=0; sample<temp_x.Rows(); sample++)
              */
                {
                   //printf("outer alpha =(%dx%d) y_outer =(%dx%d)",model_alpha.Outer(model_alpha).Rows(),model_alpha.Outer(model_alpha).Cols(),Y.Outer(Y).Rows(),Y.Outer(Y).Cols());
                   
                   gradient = ones - (model_alpha.Outer(model_alpha) * Y.Outer(Y) * this.kernel.KernelFunction(X, X)).Sum();
                          
                   model_alpha += config.alpha * gradient;
                   model_alpha.Clip(0, INT_MAX);
               }
           }
           
        //--- Print the loss at the end of an epoch
       
         is_fitted_already = true;  
         
         //preds = this.predict(X);
         
         loss[epoch] = preds.Loss(this.Y, LOSS_BCE);
        
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
