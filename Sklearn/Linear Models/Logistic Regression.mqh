//+------------------------------------------------------------------+
//|                                                 MatrixExtend::mqh |
//|                                  Copyright 2022, Omega Joctan  . |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <MALE5\Utils.mqh>
#include <MALE5\Sklearn\metrics.mqh>
#include <MALE5\Numpy\Numpy.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CLogisticRegression
  {
private:
   CNumpy            np;
   vector            classes_in_data;

   bool              istrained;
   bool              checkIsTrained(string func)
     {
      if(!istrained)
        {
         Print(func," Tree not trained, Call fit function first to train the model");
         return false;
        }
      return (true);
     }

   bool              CheckSamplesSize(string func, ulong size)
     {
      if(size != m_features)
        {
         printf("%s x sample size doesn't align with the training data m_features %d",func, size);
         return false;
        }
      return true;
     }

   matrix            weights;
   double            bias;

   //---

   uint              m_epochs;
   double            m_alpha;
   double            m_tol;
   ulong             m_features;
   int               m_random_seed;

public:
                     CLogisticRegression(uint epochs=10, double alpha=0.01, double tol=1e-8, int random_seed = 0);
                    ~CLogisticRegression(void);


   void              fit(matrix &x, vector &y);

   int               predict(vector &x);
   vector            predict(matrix &x);
   double            predict_proba(vector &x);
   vector            predict_proba(matrix &x);

  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::CLogisticRegression(uint epochs=10, double alpha=0.01, double tol=1e-8, int random_seed=0)
   :istrained(false),
    m_epochs(epochs),
    m_alpha(alpha),
    m_tol(tol),
    m_random_seed(random_seed)
  {

  }
//+------------------------------------------------------------------+
//| This is where the logistic model gets trained                    |
//+------------------------------------------------------------------+
void CLogisticRegression::fit(matrix &x, vector &y)
  {
   ulong m = x.Rows(), n = x.Cols();
   m_features = n;
   
   np.random.seed(m_random_seed);
   vector rand_v = np.random.uniform(-1, 1, (uint)n);
   this.weights = np.expand_dims(rand_v, 1);
   
   //---
   
   matrix dw; //derivative wrt weights &
   double db; //bias respectively
   vector preds;

   istrained = true;

   double prev_cost = -DBL_MAX, cost =0;
   for(ulong i=0; i<m_epochs; i++)
     {
      preds = this.predict_proba(x);

      //-- Computing gradient(s)

      matrix error = np.expand_dims(preds - y,  1);

      dw = (1/(double)m) * x.Transpose().MatMul(error);
      db = (1/(double)m) * (preds - y).Sum();

      cost = Metrics::mse(y, preds);

      printf("---> Logistic regression build epoch [%d/%d] mse %.5f",i+1,m_epochs, cost);

      this.weights -= this.m_alpha * dw;
      this.bias -= this.bias * db;

      if(MathAbs(prev_cost - cost) < this.m_tol)
        {
         Print("Converged!!!");
         break;
        }

      prev_cost = cost;
     }
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::~CLogisticRegression(void)
  {

  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CLogisticRegression::predict(vector &x)
  {
   if(!checkIsTrained(__FUNCTION__))
      return 0;

   if(!CheckSamplesSize(__FUNCTION__,x.Size()))
      return 0;

   matrix x_mat = np.expand_dims(x, 1);
   matrix preds = (x_mat.MatMul(this.weights) + this.bias);

   preds.Activation(preds, AF_HARD_SIGMOID);

   if(preds.Rows()>1)
     {
      printf("%s The outcome from a sigmoid must be a scalar value",__FUNCTION__);
      return 0;
     }
   return (int)(preds[0][0]>=0.5);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::predict(matrix &x)
  {
   vector v(x.Rows());
   for(ulong i=0; i<x.Rows(); i++)
      v[i] = this.predict(x.Row(i));

   return v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CLogisticRegression::predict_proba(vector &x)
  {
   if(!checkIsTrained(__FUNCTION__))
      return 0;

   matrix x_mat = np.expand_dims(x, 1);
   matrix preds = (x_mat.MatMul(this.weights) + this.bias);

   preds.Activation(preds, AF_HARD_SIGMOID);

   if(preds.Rows()>1)
     {
      printf("%s The outcome from a sigmoid must be a scalar value",__FUNCTION__);
      return 0;
     }
   return preds[0][0];
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CLogisticRegression::predict_proba(matrix &x)
  {
   vector v(x.Rows());
   for(ulong i=0; i<x.Rows(); i++)
      v[i] = this.predict_proba(x.Row(i));

   return v;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
