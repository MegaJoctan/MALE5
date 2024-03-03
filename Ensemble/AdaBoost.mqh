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
//|         AdaBoost class for Decision Tree                         |
//+------------------------------------------------------------------+
namespace DecisionTree
 {
class AdaBoost
  {
  
protected:
                     vector m_alphas;
                     vector classes_in_data;
                     int m_random_state;
                     bool m_boostrapping;
                     uint m_min_split, m_max_depth;
                     
                     CDecisionTreeClassifier *weak_learners[]; //store weak_learner pointers for memory allocation tracking
                     CDecisionTreeClassifier *weak_learner;
                     
                     uint m_estimators;
                     
public:
                     AdaBoost(uint min_split, uint max_split, uint n_estimators=50, int random_state=42, bool bootstrapping=true);
                    ~AdaBoost(void);
                    
                    void fit(matrix &x, vector &y);
                    int predict(vector &x);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AdaBoost::AdaBoost(uint min_split, uint max_split, uint n_estimators=50, int random_state=42, bool bootstrapping=true)
:m_estimators(n_estimators),
 m_random_state(random_state),
 m_boostrapping(bootstrapping),
 m_min_split(min_split),
 m_max_depth(max_split)
 {
   ArrayResize(weak_learners, m_estimators);   //Resizing the array to retain the number of base weak_learners
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AdaBoost::~AdaBoost(void)
 {   
   for (uint i=0; i<m_estimators; i++) //Delete the forest | all trees
     if (CheckPointer(weak_learners[i]) != POINTER_INVALID)
      delete(weak_learners[i]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void AdaBoost::fit(matrix &x,vector &y)
 {
   m_alphas.Resize(m_estimators);  
   classes_in_data = MatrixExtend::Unique(y); //Find the target variables in the class
      
   ulong m = x.Rows(), n = x.Cols();
   vector weights(m); weights = weights.Fill(1.0) / m; //Initialize instance weights
   vector preds(m);
   vector misclassified(m);
   
//---

   matrix data = MatrixExtend::concatenate(x, y);
   matrix temp_data;
   
   matrix x_subset;
   vector y_subset;

   double error = 0;
   
   for (uint i=0; i<m_estimators; i++)
    {      
      temp_data = data;
      MatrixExtend::Randomize(temp_data, this.m_random_state, this.m_boostrapping);
      
       if (!MatrixExtend::XandYSplitMatrices(temp_data, x_subset, y_subset)) //Get randomized subsets
         {  
            ArrayRemove(weak_learners,i,1); //Delete the invalid weak_learner
            printf("%s %d Failed to split data",__FUNCTION__,__LINE__);
            continue;
         }

//---
      
      weak_learner = new CDecisionTreeClassifier(this.m_min_split, m_max_depth);             
              
      weak_learner.fit(x_subset, y_subset); //fiting the randomized data to the i-th weak_learner
      preds = weak_learner.predict(x_subset); //making predictions for the i-th weak_learner
             
       for (ulong j=0; j<m; j++)
          misclassified[j] = (preds[j] != y_subset[j]);
       
       error = (misclassified * weights).Sum() / (double)weights.Sum();
       
      //--- Calculate the weight of a weak learner in the final weak_learner
      
      double alpha = 0.5 * log((1-error) / (error + 1e-10));
      
      //--- Update instance weights
      
      weights *= exp(-alpha * y_subset * preds);
      weights /= weights.Sum();
      
      //--- save a weak learner and its weight
      
      this.m_alphas[i] = alpha;
      this.weak_learners[i] = weak_learner;
      
      printf("Building Estimator [%d/%d] Accuracy Score %.3f",i+1,m_estimators,Metrics::accuracy_score(y_subset,preds));
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int AdaBoost::predict(vector &x)
 {
   // Combine weak learners using weighted sum   
   
   vector weak_preds(m_estimators), 
          final_preds(m_estimators);
          
   for (uint i=0; i<this.m_estimators; i++)
     weak_preds[i] = this.weak_learners[i].predict(x);
  
  return (int)weak_preds[(this.m_alphas*weak_preds).ArgMax()]; //Majority decision
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector AdaBoost::predict(matrix &x)
 {
   vector ret_v(x.Rows());
   for (ulong i=0; i<ret_v.Size(); i++)
      ret_v[i] = this.predict(x.Row(i));
   
   return ret_v;
 }
}

//+------------------------------------------------------------------+
//|         Adaboost for Logistic Regression                         |
//+------------------------------------------------------------------+

namespace LogisticRegression
 {
class AdaBoost
  {
  
protected:
                     vector m_alphas;
                     vector classes_in_data;
                     int m_random_state;
                     bool m_boostrapping;
                     uint m_min_split, m_max_depth;
                     
                     CLogisticRegression *weak_learners[]; //store weak_learner pointers for memory allocation tracking
                     CLogisticRegression *weak_learner;
                     
                     uint m_estimators;
                     
public:
                     AdaBoost(uint n_estimators=50, int random_state=42, bool bootstrapping=true);
                    ~AdaBoost(void);
                    
                    void fit(matrix &x, vector &y);
                    int predict(vector &x);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AdaBoost::AdaBoost(uint n_estimators=50, int random_state=42, bool bootstrapping=true)
:m_estimators(n_estimators),
 m_random_state(random_state),
 m_boostrapping(bootstrapping)
 {
   ArrayResize(weak_learners, m_estimators);   //Resizing the array to retain the number of base weak_learners
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
AdaBoost::~AdaBoost(void)
 {   
   for (uint i=0; i<m_estimators; i++) //Delete the forest | all trees
     if (CheckPointer(weak_learners[i]) != POINTER_INVALID)
      delete(weak_learners[i]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void AdaBoost::fit(matrix &x,vector &y)
 {
   m_alphas.Resize(m_estimators);  
   classes_in_data = MatrixExtend::Unique(y); //Find the target variables in the class
      
   ulong m = x.Rows(), n = x.Cols();
   vector weights(m); weights = weights.Fill(1.0) / m; //Initialize instance weights
   vector preds(m);
   vector misclassified(m);
   
//---

   matrix data = MatrixExtend::concatenate(x, y);
   matrix temp_data;
   
   matrix x_subset;
   vector y_subset;

   double error = 0;
   
   for (uint i=0; i<m_estimators; i++)
    {      
      
      temp_data = data;
      MatrixExtend::Randomize(temp_data, this.m_random_state, this.m_boostrapping);
      
       if (!MatrixExtend::XandYSplitMatrices(temp_data, x_subset, y_subset)) //Get randomized subsets
         {  
            ArrayRemove(weak_learners,i,1); //Delete the invalid weak_learner
            printf("%s %d Failed to split data",__FUNCTION__,__LINE__);
            continue;
         }

//---
      
      weak_learner = new CLogisticRegression();             
              
      weak_learner.fit(x_subset, y_subset); //fiting the randomized data to the i-th weak_learner
      preds = weak_learner.predict(x_subset); //making predictions for the i-th weak_learner
             
       for (ulong j=0; j<m; j++)
          misclassified[j] = (preds[j] != y_subset[j]);
       
       error = (misclassified * weights).Sum() / (double)weights.Sum();
       
      //--- Calculate the weight of a weak learner in the final weak_learner
      
      double alpha = 0.5 * log((1-error) / (error + 1e-10));
      
      //--- Update instance weights
      
      weights *= exp(-alpha * y_subset * preds);
      weights /= weights.Sum();
      
      //--- save a weak learner and its weight
      
      this.m_alphas[i] = alpha;
      this.weak_learners[i] = weak_learner;
      
      printf("Building Estimator [%d/%d] Accuracy Score %.3f",i+1,m_estimators,Metrics::accuracy_score(y_subset,preds));
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int AdaBoost::predict(vector &x)
 {
   // Combine weak learners using weighted sum   
   
   vector weak_preds(m_estimators), 
          final_preds(m_estimators);
          
   for (uint i=0; i<this.m_estimators; i++)
     weak_preds[i] = this.weak_learners[i].predict(x);
  
  return (int)weak_preds[(this.m_alphas*weak_preds).ArgMax()]; //Majority decision
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector AdaBoost::predict(matrix &x)
 {
   vector ret_v(x.Rows());
   for (ulong i=0; i<ret_v.Size(); i++)
      ret_v[i] = this.predict(x.Row(i));
   
   return ret_v;
 }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
