//+------------------------------------------------------------------+
//|                                                       forest.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <MALE5\Decision Tree\tree.mqh>
#include <MALE5\metrics.mqh>

#define Random(mini, maxi) mini + int((MathRand() / 32767.0) * (maxi - mini))

class CRandomForestClassifier
  {
CMetrics metrics;

protected:
   uint  m_ntrees;
   uint  m_maxdepth;
   uint  m_minsplit;
   int   m_random_state;
   
   CMatrixutils matrix_utils;
   CDecisionTreeClassifier *forest[];
   string ConvertTime(double seconds);
   
public:
                     CRandomForestClassifier(uint n_trees=100, uint minsplit=NULL, uint max_depth=NULL, int random_state=-1);
                    ~CRandomForestClassifier(void);
                    
                    void fit(matrix &x, vector &y, bool replace=true);
                    double predict(vector &x);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRandomForestClassifier::CRandomForestClassifier(uint n_trees=100, uint minsplit=NULL, uint max_depth=NULL, int random_state=-1):
   m_ntrees(n_trees),
   m_maxdepth(max_depth),
   m_minsplit(minsplit),
   m_random_state(random_state)
 {
   
   ArrayResize(forest, n_trees);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRandomForestClassifier::~CRandomForestClassifier(void)
 {
   for (uint i=0; i<m_ntrees; i++) //Delete the forest | all trees
     if (CheckPointer(forest[i]) != POINTER_INVALID)
      delete(forest[i]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRandomForestClassifier::fit(matrix &x, vector &y, bool replace=true)
 {
  matrix x_subset;
  vector y_subset;
  matrix data = this.matrix_utils.concatenate(x, y, 1);
  matrix temp_data = data;
  vector preds;
  
  datetime time_start = GetTickCount(), current_time;
  
  Print("[ Classifier Random Forest Building ]");
    
   for (uint i=0; i<m_ntrees; i++)
     {
       time_start = GetTickCount();
       
       printf("   ===> Tree no: %d",i+1);
       
       temp_data = data;
       matrix_utils.Randomize(temp_data, m_random_state, replace);
       
       if (!this.matrix_utils.XandYSplitMatrices(temp_data, x_subset, y_subset)) //Get randomized subsets
         {
            ArrayRemove(forest,i,1); //Delete the invalid tree in a forest
            printf("%s %d Failed to split data for a tree ",__FUNCTION__,__LINE__);
            continue;
         } 
       
       forest[i] = new CDecisionTreeClassifier(this.m_minsplit, this.m_maxdepth);
              
       forest[i].fit(x_subset, y_subset); //Add the trained tree to the forest
       preds = forest[i].predict(x_subset);
       
       current_time = GetTickCount();
       
       printf("       Accuracy Score: %.3f Time taken %s",metrics.accuracy_score(y_subset, preds),ConvertTime((current_time - time_start) / 1000.0));
     }
     
   m_ntrees = ArraySize(forest); //The successfully build trees
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CRandomForestClassifier::predict(vector &x)
 {
   vector predictions(m_ntrees); //predictions from all the trees
    
    for (uint i=0; i<this.m_ntrees; i++) //all trees make the predictions
      predictions[i] = forest[i].predict(x);
      
   vector uniques = matrix_utils.Unique(predictions);   
   
   return uniques[matrix_utils.Unique_count(predictions).ArgMax()]; //select the majority decision
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CRandomForestClassifier::predict(matrix &x)
 {
   vector preds(x.Rows());
   
   for (ulong i=0; i<x.Rows(); i++)
     preds[i] = this.predict(x.Row(i));
  
  return preds;     
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CRandomForestClassifier::ConvertTime(double seconds)
{
    string time_str = "";
    uint minutes = 0, hours = 0;

    if (seconds >= 60)
    {
        minutes = (uint)(seconds / 60.0) ;
        seconds = fmod(seconds, 1.0) * 60;
        time_str = StringFormat("%d Minutes and %.3f Seconds", minutes, seconds);
    }
    
    if (minutes >= 60)
    {
        hours = (uint)(minutes / 60.0);
        minutes = minutes % 60;
        time_str = StringFormat("%d Hours and %d Minutes", hours, minutes);
    }

    if (time_str == "")
    {
        time_str = StringFormat("%.3f Seconds", seconds);
    }

    return time_str;
}
 
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//|      Random Forest for regression problems                       |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

class CRandomForestRegressor
  {
CMetrics metrics;

private:
   uint  m_ntrees;
   uint  m_maxdepth;
   uint  m_minsplit;
   int   m_random_state;
   
   CMatrixutils matrix_utils;
   CDecisionTreeRegressor *forest[];
   
   string ConvertTime(double seconds);
   
public:
                     CRandomForestRegressor(uint n_trees=100, uint minsplit=NULL, uint max_depth=NULL, int random_state=-1);
                    ~CRandomForestRegressor(void);
                    
                    void fit(matrix &x, vector &y, bool replace=true);
                    double predict(vector &x);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRandomForestRegressor::CRandomForestRegressor(uint n_trees=100, uint minsplit=NULL, uint max_depth=NULL, int random_state=-1):
   m_ntrees(n_trees),
   m_maxdepth(max_depth),
   m_minsplit(minsplit),
   m_random_state(random_state)
 {
   
   ArrayResize(forest, n_trees);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRandomForestRegressor::~CRandomForestRegressor(void)
 {
   for (uint i=0; i<m_ntrees; i++) //Delete the forest | all trees
     if (CheckPointer(forest[i]) != POINTER_INVALID)
      delete(forest[i]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CRandomForestRegressor::ConvertTime(double seconds)
{
    string time_str = "";
    uint minutes = 0, hours = 0;

    if (seconds >= 60)
    {
        minutes = (uint)(seconds / 60.0) ;
        seconds = fmod(seconds, 1.0) * 60;
        time_str = StringFormat("%d Minutes and %.3f Seconds", minutes, seconds);
    }
    
    if (minutes >= 60)
    {
        hours = (uint)(minutes / 60.0);
        minutes = minutes % 60;
        time_str = StringFormat("%d Hours and %d Minutes", hours, minutes);
    }

    if (time_str == "")
    {
        time_str = StringFormat("%.3f Seconds", seconds);
    }

    return time_str;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRandomForestRegressor::fit(matrix &x, vector &y, bool replace=true)
 {
  matrix x_subset;
  vector y_subset;
  matrix data = this.matrix_utils.concatenate(x, y, 1);
  matrix temp_data = data;
  
  vector preds;
  
  datetime time_start = GetTickCount(), current_time;
  
  Print("[ Regressor Random Forest Building ]");
    
   for (uint i=0; i<m_ntrees; i++)
     {
       time_start = GetTickCount();
       
       temp_data = data;
       matrix_utils.Randomize(temp_data, m_random_state, replace);
       
       if (!this.matrix_utils.XandYSplitMatrices(temp_data, x_subset, y_subset)) //Get randomized subsets
         {  
            ArrayRemove(forest,i,1); //Delete the invalid tree in a forest
            printf("%s %d Failed to split data for a tree ",__FUNCTION__,__LINE__);
            continue;
         }
       
       forest[i] = new CDecisionTreeRegressor(this.m_minsplit, this.m_maxdepth);       
       forest[i].fit(x_subset, y_subset); //Add the trained tree to the forest
       preds = forest[i].predict(x_subset);
       
       current_time = GetTickCount();
       printf("   ===> Tree no: %d R_2 Score: %.3f Time taken: %s",i+1,metrics.r_squared(y_subset, preds), ConvertTime((current_time - time_start) / 1000.0));
     }
     
   m_ntrees = ArraySize(forest); //The successfully build trees  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CRandomForestRegressor::predict(vector &x)
 {
    vector predictions(m_ntrees); //predictions from all the trees
    
    for (uint i=0; i<this.m_ntrees; i++)
      predictions[i] = forest[i].predict(x);

   return predictions.Mean();   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CRandomForestRegressor::predict(matrix &x)
 {
   vector preds(x.Rows());
   
   for (ulong i=0; i<x.Rows(); i++)
     preds[i] = this.predict(x.Row(i));
  
  return preds;     
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
