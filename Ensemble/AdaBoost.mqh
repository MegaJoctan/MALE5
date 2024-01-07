//+------------------------------------------------------------------+
//|                                                     AdaBoost.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

#include <MALE5\MatrixExtend.mqh>
#include <MALE5\Naive Bayes\Naive Bayes.mqh>

//+------------------------------------------------------------------+
//|      Model class                                                 |
//+------------------------------------------------------------------+

#include <MALE5\Decision Tree\tree.mqh>
#include <MALE5\Linear Models\Logistic Regression.mqh>

//+------------------------------------------------------------------+
//|      AdaBoost class                                              |
//+------------------------------------------------------------------+
namespace CAdaBoostDecisionTree
 {
class CAdaBoost
  {
  
protected:
                     vector m_alphas;
                     matrix m_weights;
                     
                     CDecisionTreeClassifier models[]; //store model pointers for memory allocation tracking
                     CDecisionTreeClassifier model;
                     
                     uint m_estimators;
                     
public:
                     CAdaBoost(CDecisionTreeClassifier &base_model, uint n_estimators=50, int random_state=42);
                    ~CAdaBoost(void);
                    
                    void fit(matrix &x, vector &y);
                    int predict(vector &x);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAdaBoost::CAdaBoost(CDecisionTreeClassifier &base_model, uint n_estimators=50, int random_state=42)
:m_estimators(n_estimators)
 {
   ArrayResize(models, n_estimators);   
   model = base_model;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAdaBoost::~CAdaBoost(void)
 {   
   //for (uint i=0; i<models.Size(); i++)
   //  if (CheckPointer(models[i]) != POINTER_INVALID)
   //    delete models[i];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CAdaBoost::fit(matrix &x,vector &y)
 {
   m_alphas.Resize(y.Size(), m_estimators);  
   m_weights.Resize(y.Size(), m_estimators);  
   
   
   ulong m = x.Rows(), n = x.Cols();
   vector weights(m); weights = weights.Fill(1.0) / m; //Initialize instance weights
   vector preds(m);
   vector misclassified(m);
   
   double error = 0;
   
   for (uint i=0; i<m_estimators; i++)
    {      
      model.fit(x, y);
      preds = model.predict(x);
       
       for (ulong j=0; j<m; j++) misclassified[j] = (preds[j] != y[j]);
       
       error = (misclassified * weights).Sum() / (double)weights.Sum();
       
      //--- Calculate the weight of a weak learner in the final model
      
      double alpha = 0.5 * log((1-error) / (error + 1e-10));
      
      //--- Update instance weights
      
      weights *= exp(-alpha * y * preds);
      weights /= weights.Sum();
      
      //--- save a weak learner and its weight
      
      this.m_alphas[i] = alpha;
      this.models[i] = model;
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CAdaBoost::predict(vector &x)
 {
   // Combine weak learners using weighted sum   
   
   vector weak_preds(m_estimators), 
          final_preds(m_estimators);
          
   for (uint i=0; i<this.m_estimators; i++)
     weak_preds[i] = this.models[i].predict(x);
  
  final_preds = MatrixExtend::Sign(MatrixExtend::VectorToMatrix(this.m_alphas, this.m_alphas.Size()).MatMul(MatrixExtend::VectorToMatrix(weak_preds, weak_preds.Size())));
  
  if (MQLInfoInteger(MQL_DEBUG))
   Print("Final preds: ",final_preds); 
         
  return (int)final_preds[0];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CAdaBoost::predict(matrix &x)
 {
   vector ret_v(x.Rows());
   for (ulong i=0; i<ret_v.Size(); i++)
      ret_v[i] = this.predict(x.Row(i));
   
   return ret_v;
 }
}

namespace CAdaBoostLogReg
 {
 
class CAdaBoost
  {
  
protected:
                     vector m_alphas;
                     matrix m_weights;
                     
                     CLogisticRegression models[]; //store model pointers for memory allocation tracking
                     CLogisticRegression model;
                     
                     uint m_estimators;
                     
public:
                     CAdaBoost(CLogisticRegression &base_model, uint n_estimators=50, int random_state=42);
                    ~CAdaBoost(void);
                    
                    void fit(matrix &x, vector &y);
                    int predict(vector &x);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAdaBoost::CAdaBoost(CLogisticRegression &base_model, uint n_estimators=50, int random_state=42)
:m_estimators(n_estimators)
 {
   ArrayResize(models, n_estimators);   
   model = base_model;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAdaBoost::~CAdaBoost(void)
 {   
   //for (uint i=0; i<models.Size(); i++)
   //  if (CheckPointer(models[i]) != POINTER_INVALID)
   //    delete models[i];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CAdaBoost::fit(matrix &x,vector &y)
 {
   m_alphas.Resize(y.Size(), m_estimators);  
   m_weights.Resize(y.Size(), m_estimators);  
   
   
   ulong m = x.Rows(), n = x.Cols();
   vector weights(m); weights = weights.Fill(1.0) / m; //Initialize instance weights
   vector preds(m);
   vector misclassified(m);
   
   double error = 0;
   
   for (uint i=0; i<m_estimators; i++)
    {      
      model.fit(x, y);
      
      preds = model.predict(x);
       
       for (ulong j=0; j<m; j++) misclassified[j] = (preds[j] != y[j]);
       
       error = (misclassified * weights).Sum() / (double)weights.Sum();
       
      //--- Calculate the weight of a weak learner in the final model
      
      double alpha = 0.5 * log((1-error) / error);
      
      //--- Update instance weights
      
      weights *= exp(-alpha * y * preds);
      weights /= weights.Sum();
      
      //--- save a weak learner and its weight
      
      this.m_alphas[i] = alpha;
      this.models[i] = model;
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int CAdaBoost::predict(vector &x)
 {
   // Combine weak learners using weighted sum   
   
   vector weak_preds(m_estimators), 
          final_preds(m_estimators);
          
   for (uint i=0; i<this.m_estimators; i++)
     weak_preds[i] = this.models[i].predict(x);
  
  final_preds = MatrixExtend::Sign(MatrixExtend::VectorToMatrix(this.m_alphas, this.m_alphas.Size()).MatMul(weak_preds));
  
  if (MQLInfoInteger(MQL_DEBUG))
   Print("Final preds: ",final_preds); 
         
  return (int)final_preds[0];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CAdaBoost::predict(matrix &x)
 {
   vector ret_v(x.Rows());
   for (ulong i=0; i<ret_v.Size(); i++)
      ret_v[i] = this.predict(x.Row(i));
   
   return ret_v;
 }
}