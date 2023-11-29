//+------------------------------------------------------------------+
//|                                                         tree.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <MALE5\matrix_utils.mqh>

struct Node
  {
  // for decision node
  
  uint feature_index;
  double threshold;
  matrix left_child,
         right_child;
  double info_gain;
  
  // for leaf node
  
  double value;
     
     Node() {} //default constructor
     Node(uint &feature_index_, double threshold_, matrix &left_, matrix &right_, double info_gain_=NULL, double value_=NULL)
      { 
        /* constructor */
        
        // for decision node
        
        this.feature_index = feature_index_;
        this.threshold = threshold_;
        this.left_child = left_;
        this.right_child = right_;
        this.info_gain = info_gain_;
        
        // for leaf node
        
        this.value = value_;
      }
  };

#define log2(value) MathLog(value) / MathLog(2)

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CDecisionTree
  {
protected:
   
   Node get_best_split(matrix &data, uint num_features);
   Node build_tree(matrix &data, uint curr_depth=0);

//---

   CMatrixutils   matrix_utils;
   
   uint m_max_depth;
   uint m_min_samples_split;
   
   enum mode {MODE_ENTROPY, MODE_GINI};
   mode m_mode;
   
   double  gini_index(vector &y);
   double  entropy(vector &y);
   double  information_gain(vector &parent, vector &left_child, vector &right_child);
   double  calculate_leaf_value(vector &Y);
   
   void  split_data(matrix &left, matrix& right, const matrix &data, uint feature_index, double threshold=0.5);
   
     
public:
                     CDecisionTree(uint min_samples_split=2, uint max_depth=2, mode mode_=MODE_GINI);
                    ~CDecisionTree(void);
                    
                    void fit(matrix &x, vector &y);
                    
                    void build_decision_tree(matrix &data, vector &labels);
                    int predict_sample(vector &node, vector &sample);
                    vector predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDecisionTree::CDecisionTree(uint min_samples_split=2, uint max_depth=2, mode mode_=MODE_GINI)
 {
   m_min_samples_split = min_samples_split;
   m_max_depth = max_depth;
   
   m_mode = mode_;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDecisionTree::~CDecisionTree(void)
 {
   //while(CheckPointer(node)!=POINTER_INVALID)
   //  delete (node);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDecisionTree::gini_index(vector &y)
 {
   vector unique = matrix_utils.Unique(y);
   
   vector probabilities = unique / (double)y.Size();
   
   return 1.0 - MathPow(probabilities, 2).Sum();
 }
//+------------------------------------------------------------------+
//|      function to compute entropy                                 |
//+------------------------------------------------------------------+
double CDecisionTree::entropy(vector &y)
 {    
   vector class_labels = matrix_utils.Unique(y);
     
   vector p_cls = class_labels / double(y.Size());
  
   vector entropy = (-1 * p_cls) * log2(p_cls);
  
  return entropy.Sum();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDecisionTree::information_gain(vector &parent, vector &left_child, vector &right_child)
 {  
    double weight_left = left_child.Size() / (double)parent.Size(),
           weight_right = right_child.Size() / (double)parent.Size();
    
    double gain =0;
    
    switch(m_mode)
      {
       case  MODE_GINI:
         gain = gini_index(parent) - ( (weight_left*gini_index(left_child)) + (weight_right*gini_index(right_child)) );
         break;
       case MODE_ENTROPY:
         gain = entropy(parent) - ( (weight_left*entropy(left_child)) + (weight_right*entropy(right_child)) );
         break;
      }
    
   return gain;
 }
//+------------------------------------------------------------------+
//|         function to print the tree                               |
//+------------------------------------------------------------------+
/*
void CDecisionTree::print_tree(vector tree=NULL, string indent=" "):
  {
     if (!tree)
         tree = this.root;

     if tree.value is not None:
         print(tree.value)

     else:
         print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
         print("%sleft:" % (indent), end="")
         self.print_tree(tree.left, indent + indent)
         print("%sright:" % (indent), end="")
         self.print_tree(tree.right, indent + indent)
  }         
*/
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
/*
void CDecisionTree::build_decision_tree(matrix &data, vector &labels)
 {
  if (m_max_depth == 0 || labels.Size() == 1)
      return CDecisionTree(m_max_depth, m_treshold);

  current_gini = this.gini_index(labels);
  
  double best_gain = 0.0;
  bool best_criteria = false;
  bool best_sets = false;

  num_features = data.shape[1]

  for (feature_index in range(num_features))
    {
      feature_values = np.unique(data[:, feature_index])

      for (threshold in feature_values)
        {
          left_data, left_labels, right_data, right_labels = this.split_data(data, labels, feature_index, threshold)
          if (len(left_data) > 0 and len(right_data) > 0)
            {
              gain = this.information_gain(left_labels, right_labels, current_gini);
              
              if (gain > best_gain)
                {
                  best_gain = gain;
                  best_criteria = (feature_index, threshold);
                  best_sets = (left_data, left_labels, right_data, right_labels);
                }
            }
        }
     }
            
  if (best_gain > 0)
    {
      left_branch = this.build_decision_tree(best_sets[0], best_sets[1], depth - 1);
      right_branch = this.build_decision_tree(best_sets[2], best_sets[3], depth - 1);
      
      return CDecisionTree(m_max_depth, m_treshold);
    }
  else:
      return DecisionTreeNode(predicted_class=np.argmax(np.bincount(labels)));
 }*/
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDecisionTree::fit(matrix &x, vector &y)
 {
   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDecisionTree::split_data(matrix &left, matrix& right, const matrix &data, uint feature_index, double threshold=0.5)
 {
   int left_size=0, right_size =0;
   vector row = {};
   
   left.Resize(0, data.Cols());
   right.Resize(0,data.Cols());
   
    for (ulong i=0; i<data.Rows(); i++)
     {       
       row = data.Row(i);
       
       if (row[feature_index] <= threshold)
        {
          left_size++;
          left.Resize(left_size, left.Cols());
          left.Row(row, left_size-1); 
        }
       else
        {
         right_size++;
         right.Resize(right_size, right.Cols());
         right.Row(row, right_size-1);
        }
     }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
/*
void CDecisionTree::get_best_split(matrix &data, uint num_features)
 {
   vector best_split = {};
   double max_info_gain = -DBL_MAX;
   vector feature_values = {};
   
   matrix left, right;
   vector left_v, right_v, y_v;
   
   
   for (uint i=0; i<num_features; i++)
     {
       feature_values = data.Row(i);
       vector possible_thresholds = matrix_utils.Unique(feature_values);
         
         for (uint j=0; j<possible_thresholds.Size(); j++)
            {
              this.split_data(left, right, data, i, possible_thresholds[j]);
              
              if (left.Rows()>0 && right.Rows() > 0)
                {
                  y_v = data.Col(data.Cols()-1);
                  right_v = right.Col(right.Cols()-1);
                  left_v = left.Col(left.Cols()-1);
                  
                  double curr_info_gain = this.information_gain(y_v, left_v, right_v);
                  
                  if (curr_info_gain > max_info_gain)
                    {
                     //best_split
                    }
                }
            }    
     }
     
 } */
//+------------------------------------------------------------------+
//|      Return the Node for the best split                          |
//+------------------------------------------------------------------+
Node CDecisionTree::get_best_split(matrix &data, uint num_features)
  {
    uint feature_index_;
    double threshold_=0; 
    matrix left, right;
    double info_gain_=NULL, value_=NULL;
    
    
   double max_info_gain = -DBL_MAX;
   vector feature_values = {};
   
   vector left_v, right_v, y_v;
   
//---

   for (uint i=0; i<num_features; i++)
     {
       feature_values = data.Row(i);
       vector possible_thresholds = matrix_utils.Unique(feature_values);
         
         for (uint j=0; j<possible_thresholds.Size(); j++)
            {
              this.split_data(left, right, data, i, possible_thresholds[j]);
              
              if (left.Rows()>0 && right.Rows() > 0)
                {
                  y_v = data.Col(data.Cols()-1);
                  right_v = right.Col(right.Cols()-1);
                  left_v = left.Col(left.Cols()-1);
                  
                  double curr_info_gain = this.information_gain(y_v, left_v, right_v);
                  
                  if (curr_info_gain > max_info_gain)
                    {
                      feature_index_ = i;
                      threshold_ = possible_thresholds[j];
                      info_gain_ = curr_info_gain;
                      
                      max_info_gain = curr_info_gain;
                    }
                }
            }    
     }
     
    return Node(feature_index_, threshold_, left, right, info_gain_);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
Node CDecisionTree::build_tree(matrix &data, uint curr_depth=0)
 {
    matrix X;
    vector Y;
      
    matrix_utils.XandYSplitMatrices(data,X,Y); 
    
    uint feature_index_;
    double threshold_=0; 
    matrix left, right;
    double info_gain_=NULL, value_=NULL;
    
    ulong samples = X.Rows(), features = X.Cols();
    
    if (samples >= m_min_samples_split && curr_depth<=m_max_depth)
      {
         Node best_split = this.get_best_split(data, (uint)features);
         
         if (best_split.info_gain > 0)
           {
             Node left_subtree = this.build_tree(best_split.left_child, curr_depth+1);
             Node right_subtree = this.build_tree(best_split.right_child, curr_depth+1);
             
             feature_index_ = best_split.feature_index;
             threshold_ = best_split.threshold;
             left = best_split.left_child;
             right = best_split.right_child;
             info_gain_ = best_split.info_gain;
             
           }
      }      
      
     double leaf_value = this.calculate_leaf_value(Y);
     return Node(feature_index_,threshold_,left,right,info_gain_,leaf_value);
 }
//+------------------------------------------------------------------+
//|   returns the element from Y that has the highest count,         |
//|  effectively finding the most common element in the list.        |
//+------------------------------------------------------------------+
double CDecisionTree::calculate_leaf_value(vector &Y)
 {   
   vector uniques = matrix_utils.Unique(Y);
   vector classes = matrix_utils.Classes(Y);
   
   return classes[uniques.ArgMax()];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

 