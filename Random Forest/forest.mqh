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

class CRandomForest: protected CDecisionTree
  {
CMetrics metrics;

private:
   uint  m_ntrees;
   uint  m_maxdepth;
   uint  m_minsplit;
   int   m_random_state;
   
   CDecisionTree *forest[];
   
public:
                     CRandomForest(uint n_trees=100, uint minsplit=NULL, uint max_depth=NULL, int random_state=-1);
                    ~CRandomForest(void);
                    
                    void fit(matrix &x, vector &y);
                    double predict(vector &x);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CRandomForest::CRandomForest(uint n_trees=100, uint minsplit=NULL, uint max_depth=NULL, int random_state=-1):
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
CRandomForest::~CRandomForest(void)
 {
   for (uint i=0; i<m_ntrees; i++) //Delete the forest | all trees
     if (CheckPointer(forest[i]) != POINTER_INVALID)
      delete(forest[i]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CRandomForest::fit(matrix &x, vector &y)
 {
  matrix x_subset;
  vector y_subset;
  matrix data = this.matrix_utils.concatenate(x, y, 1);
  
  CDecisionTree *tree;
  
  Print("[ Random Forest Building ]");
    
   for (uint i=0; i<m_ntrees; i++)
     {
       tree = new CDecisionTree(this.m_minsplit, this.m_maxdepth);
       
       matrix_utils.Randomize(data, m_random_state);
       this.matrix_utils.XandYSplitMatrices(data, x_subset, y_subset); //Get randomized subsets
       
       tree.fit(x_subset, y_subset);
       vector preds = tree.predict(x_subset);
       
       printf("   ===> Tree no: %d Accuracy Score: %.3f ",i+1,metrics.accuracy_score(y_subset, preds));
       
       forest[i] = tree; //Add the trained tree to the forest
      
       delete (tree); //delete that tree to prevent memory leaks
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CRandomForest::predict(vector &x)
 {
    vector predictions(m_ntrees); //predictions from all the trees
    
    for (uint i=0; i<this.m_ntrees; i++)
      predictions[i] = forest[i].predict(x);
      
   
   return round(predictions.Mean());   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CRandomForest::predict(matrix &x)
 {
   vector preds(x.Rows());
   
   for (ulong i=0; i<x.Rows(); i++)
     preds[i] = this.predict(x.Row(i));
  
  return preds;     
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

