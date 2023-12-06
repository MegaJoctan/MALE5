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

struct NodeClass
  {
  };

#define log2(value) MathLog(value) / MathLog(2)

class Node
{
  public:
    // for decision node
       
    uint feature_index;
    double threshold;
    double info_gain;
     
    // for leaf node
     
    double value;   
      
    Node *left_child;
    Node *right_child;

    Node() : left_child(NULL), right_child(NULL) {} // default constructor

    Node(uint feature_index_, double threshold_=NULL, Node *left_=NULL, Node *right_=NULL, double info_gain_=NULL, double value_=NULL)
        : left_child(left_), right_child(right_)
    {
        this.feature_index = feature_index_;
        this.threshold = threshold_;
        this.info_gain = info_gain_;
        this.value = value_;
    }
    
   void Print()
    {
      printf("feature_index: %d \nthreshold: %f \ninfo_gain: %f \nleaf_value: %f",feature_index,threshold, info_gain, value);
    }    
};

struct split_info
  {
   uint feature_index;
   double threshold;
   matrix dataset_left,
          dataset_right;
   double info_gain;
  };

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
enum mode {MODE_ENTROPY, MODE_GINI};

class CDecisionTree
  {
CMatrixutils   matrix_utils;

protected:
      
   Node *build_tree(matrix &data, uint curr_depth=0);
   double  calculate_leaf_value(vector &Y);
   
//---
   
   uint m_max_depth;
   uint m_min_samples_split;
   
   mode m_mode;
   
   double  gini_index(vector &y);
   double  entropy(vector &y);
   double  information_gain(vector &parent, vector &left_child, vector &right_child);
   
   
   split_info  get_best_split(matrix &data, uint num_features);
   split_info  split_data(const matrix &data, uint feature_index, double threshold=0.5);
   
   double make_predictions(vector &x, const Node &tree);
   void delete_tree(Node* node);
   
public:
                     Node *root;
                     
                     CDecisionTree(uint min_samples_split=2, uint max_depth=2, mode mode_=MODE_GINI);
                    ~CDecisionTree(void);
                    
                     void fit(matrix &x, vector &y);
                     void print_tree(Node *tree, string indent=" ",string padl="");
                     double predict(vector &x);
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
   root = new Node();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDecisionTree::~CDecisionTree(void)
 {   
   this.delete_tree(root);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDecisionTree::delete_tree(Node* node)
 {
    if (CheckPointer(node) != POINTER_INVALID)
    {
        delete_tree(node.left_child);
        delete_tree(node.right_child);
        delete node;
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDecisionTree::gini_index(vector &y)
 {
   vector unique = matrix_utils.Unique_count(y);
   
   vector probabilities = unique / (double)y.Size();
   
   return 1.0 - MathPow(probabilities, 2).Sum();
 }
//+------------------------------------------------------------------+
//|      function to compute entropy                                 |
//+------------------------------------------------------------------+
double CDecisionTree::entropy(vector &y)
 {    
   vector class_labels = matrix_utils.Unique_count(y);
     
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
void CDecisionTree::print_tree(Node *tree, string indent=" ",string padl="")
  {
     if (tree.value != NULL)
        Print(padl+(indent=="left"?"left: ":"right: "),tree.value);
     else
       {
         indent = "";
         padl += " ";
         
         Print(padl+"X_",tree.feature_index, "<=", tree.threshold, "?", tree.info_gain);
         
         print_tree(tree.left_child, "left","--->"+padl);
         
         print_tree(tree.right_child, "right","--->"+padl);
       }
  }  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDecisionTree::fit(matrix &x, vector &y)
 {
   matrix data = matrix_utils.concatenate(x, y, 1);
   
   this.root = this.build_tree(data);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
split_info CDecisionTree::split_data(const matrix &data, uint feature_index, double threshold=0.5)
 {
   int left_size=0, right_size =0;
   vector row = {};
   
   split_info split;
   
   ulong cols = data.Cols();
   
   split.dataset_left.Resize(0, cols);
   split.dataset_right.Resize(0, cols);
      
    for (ulong i=0; i<data.Rows(); i++)
     {       
       row = data.Row(i);
       
       if (row[feature_index] <= threshold)
        {
          left_size++;
          split.dataset_left.Resize(left_size, cols);
          split.dataset_left.Row(row, left_size-1); 
        }
       else
        {
         right_size++;
         split.dataset_right.Resize(right_size, cols);
         split.dataset_right.Row(row, right_size-1);         
        }
     }
   return split;
 }
//+------------------------------------------------------------------+
//|      Return the Node for the best split                          |
//+------------------------------------------------------------------+
split_info CDecisionTree::get_best_split(matrix &data, uint num_features)
  {
  
   double max_info_gain = -DBL_MAX;
   vector feature_values = {};
   vector left_v={}, right_v={}, y_v={};
   
//---
   
   split_info best_split;
   split_info split;
   
   for (uint i=0; i<num_features; i++)
     {
       feature_values = data.Col(i);
       vector possible_thresholds = matrix_utils.Unique(feature_values);
                  
         for (uint j=0; j<possible_thresholds.Size(); j++)
            {              
              split = this.split_data(data, i, possible_thresholds[j]);
              
              if (split.dataset_left.Rows()>0 && split.dataset_right.Rows() > 0)
                {
                  y_v = data.Col(data.Cols()-1);
                  right_v = split.dataset_right.Col(split.dataset_right.Cols()-1);
                  left_v = split.dataset_left.Col(split.dataset_left.Cols()-1);
                  
                  double curr_info_gain = this.information_gain(y_v, left_v, right_v);
                                    
                  if (curr_info_gain > max_info_gain)
                    {             
                      #ifdef DEBUG_MODE
                        printf("split left: [%dx%d] split right: [%dx%d] curr_info_gain: %f max_info_gain: %f",split.dataset_left.Rows(),split.dataset_left.Cols(),split.dataset_right.Rows(),split.dataset_right.Cols(),curr_info_gain,max_info_gain);
                      #endif 
                      
                      best_split.feature_index = i;
                      best_split.threshold = possible_thresholds[j];
                      best_split.dataset_left = split.dataset_left;
                      best_split.dataset_right = split.dataset_right;
                      best_split.info_gain = curr_info_gain;
                      
                      max_info_gain = curr_info_gain;
                    }
                }
            }    
     }
     
    return best_split;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
Node *CDecisionTree::build_tree(matrix &data, uint curr_depth=0)
 {
    matrix X;
    vector Y;
      
    matrix_utils.XandYSplitMatrices(data,X,Y); 
    
    ulong samples = X.Rows(), features = X.Cols();
        
    Node *node= NULL; 
            
    if (samples >= m_min_samples_split && curr_depth<=m_max_depth)
      {
         split_info best_split = this.get_best_split(data, (uint)features);
         
         #ifdef DEBUG_MODE
          Print("best_split left: [",best_split.dataset_left.Rows(),"x",best_split.dataset_left.Cols(),"]\nbest_split right: [",best_split.dataset_right.Rows(),"x",best_split.dataset_right.Cols(),"]\nfeature_index: ",best_split.feature_index,"\nInfo gain: ",best_split.info_gain,"\nThreshold: ",best_split.threshold);
         #endif 
                  
         if (best_split.info_gain > 0)
           {
             Node *left_child = this.build_tree(best_split.dataset_left, curr_depth+1);
             Node *right_child = this.build_tree(best_split.dataset_right, curr_depth+1);
                          
             node = new Node(best_split.feature_index,best_split.threshold,left_child,right_child,best_split.info_gain);
             return node;
           }
      }      
     
     node = new Node();
     node.value = this.calculate_leaf_value(Y);
          
     return node;
 }
//+------------------------------------------------------------------+
//|   returns the element from Y that has the highest count,         |
//|  effectively finding the most common element in the list.        |
//+------------------------------------------------------------------+
double CDecisionTree::calculate_leaf_value(vector &Y)
 {   
   vector uniques = matrix_utils.Unique_count(Y);
   vector classes = matrix_utils.Unique(Y);
   
   return classes[uniques.ArgMax()];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDecisionTree::make_predictions(vector &x, const Node &tree)
 {
    if (tree.value != NULL) //This is a leaf value
      return tree.value;
      
    double feature_value = x[tree.feature_index];
    double pred = 0;
    
    #ifdef DEBUG_MODE
      Print("Tree.value: ",tree.value);
      printf("Tree.threshold %f tree.feature_index %d leaf_value %f",tree.threshold,tree.feature_index,tree.value);
    #endif 
    
    if (feature_value <= tree.threshold)
      {
       pred = this.make_predictions(x, tree.left_child);  
      }
    else
     {
       pred = this.make_predictions(x, tree.right_child);
     }
     
   return pred;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDecisionTree::predict(vector &x)
 {     
   return this.make_predictions(x, this.root);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDecisionTree::predict(matrix &x)
 {
    vector ret(x.Rows());
    
    for (ulong i=0; i<x.Rows(); i++)
       ret[i] = this.predict(x.Row(i));
       
   return ret;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
