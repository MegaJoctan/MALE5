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
#include <MALE5\MatrixExtend.mqh>


#define log2(leaf_value) MathLog(leaf_value) / MathLog(2)


class Node
{
  public:
    // for decision node
       
    uint feature_index;
    double threshold;
    double info_gain;
     
    // for leaf node
     
    double leaf_value;   
      
    Node *left_child; //left child Node
    Node *right_child; //right child Node

    Node() : left_child(NULL), right_child(NULL) {} // default constructor

    Node(int feature_index_, double threshold_=0.0, Node *left_=NULL, Node *right_=NULL, double info_gain_=NULL, double value_=NULL)
        : left_child(left_), right_child(right_)
    {
        this.feature_index = feature_index_;
        this.threshold = threshold_;
        this.info_gain = info_gain_;
        this.leaf_value = value_;
    }
    
   void __Print__()
    {
      printf("feature_index: %d \nthreshold: %f \ninfo_gain: %f \nleaf_value: %f",feature_index,threshold, info_gain, leaf_value);
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

class CDecisionTreeClassifier
  {
protected:  
   
   Node *build_tree(matrix &data, uint curr_depth=0);
   double  calculate_leaf_value(vector &Y);
   
   bool is_fitted;
   bool check_is_fitted(string func)
     {
       if (!is_fitted)
         {
           Print(func," Tree not trained, Call fit function first to train the model");
           return false;   
         }
       return (true);
     }
//---
   
   uint m_max_depth;
   uint m_min_samples_split;   
   mode m_mode;
   
   double  gini_index(vector &y);
   double  entropy(vector &y);
   double  information_gain(vector &parent, vector &l_child, vector &r_child);
   
   
   split_info  get_best_split(const matrix &data, uint num_features);
   split_info  split_data(const matrix &data, uint feature_index, double threshold=0.5);
   
   double make_predictions(const vector &x, const Node &tree);
   
   void delete_tree(Node* node);
   
   Node *nodes[]; //Keeping track of all the nodes in a tree
   
public:
                     Node *root;
                     
                     CDecisionTreeClassifier(uint min_samples_split=2, uint max_depth=2, mode mode_=MODE_GINI);
                    ~CDecisionTreeClassifier(void);
                    
                     void fit(const matrix &x, const vector &y);
                     void print_tree(Node *tree, string indent=" ",string padl="");
                     double predict(const vector &x);
                     vector predict(const matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDecisionTreeClassifier::CDecisionTreeClassifier(uint min_samples_split=2, uint max_depth=2, mode mode_=MODE_GINI)
 {
   m_min_samples_split = min_samples_split;
   m_max_depth = max_depth;
   
   m_mode = mode_;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDecisionTreeClassifier::~CDecisionTreeClassifier(void)
 {   
   #ifdef DEBUG_MODE
      Print(__FUNCTION__," Deleting Tree nodes =",nodes.Size());
   #endif 
   
   this.delete_tree(root);
   
   for (int i=0; i<(int)nodes.Size(); i++)
     this.delete_tree(nodes[i]);  
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDecisionTreeClassifier::delete_tree(Node* node)
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
double CDecisionTreeClassifier::gini_index(vector &y)
 {
   vector unique = MatrixExtend::Unique_count(y);
   
   vector probabilities = unique / (double)y.Size();
   
   return 1.0 - MathPow(probabilities, 2).Sum();
 }
//+------------------------------------------------------------------+
//|      function to compute entropy                                 |
//+------------------------------------------------------------------+
double CDecisionTreeClassifier::entropy(vector &y)
 {    
   vector class_labels = MatrixExtend::Unique_count(y);
     
   vector p_cls = class_labels / double(y.Size());
  
   vector entropy = (-1 * p_cls) * log2(p_cls);
  
  return entropy.Sum();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDecisionTreeClassifier::information_gain(vector &parent, vector &l_child, vector &r_child)
 {  
    double weight_left = l_child.Size() / (double)parent.Size(),
           weight_right = r_child.Size() / (double)parent.Size();
    
    double gain =0;    
    switch(m_mode)
      {
       case  MODE_GINI:
         gain = gini_index(parent) - ( (weight_left*gini_index(l_child)) + (weight_right*gini_index(r_child)) );
         break;
       case MODE_ENTROPY:
         gain = entropy(parent) - ( (weight_left*entropy(l_child)) + (weight_right*entropy(r_child)) );
         break;
      }
    
   return gain;
 }
//+------------------------------------------------------------------+
//|         function to print the tree                               |
//+------------------------------------------------------------------+
void CDecisionTreeClassifier::print_tree(Node *tree, string indent=" ",string padl="")
  {
     if (tree.leaf_value != NULL)
        Print((padl+indent+": "),tree.leaf_value); 
     else //if we havent' reached the leaf node keep printing child trees
       {
         padl += " ";
         
         Print((padl+indent)+": X_",tree.feature_index, "<=", tree.threshold, "?", tree.info_gain);
         
         print_tree(tree.left_child, "left","--->"+padl);
         
         print_tree(tree.right_child, "right","--->"+padl);
       }
  }  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDecisionTreeClassifier::fit(const matrix &x, const vector &y)
 {   
   matrix data = MatrixExtend::concatenate(x, y, 1);
   
   this.root = this.build_tree(data);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
split_info CDecisionTreeClassifier::split_data(const matrix &data, uint feature_index, double threshold=0.5)
 {
   int left_size=0, right_size =0;
   vector row = {};
   
   split_info split;
   
   ulong cols = data.Cols(),
         rows = data.Rows();
   
   split.dataset_left.Resize(0, cols);
   split.dataset_right.Resize(0, cols);
   
   
    for (ulong i=0; i<rows; i++)
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
split_info CDecisionTreeClassifier::get_best_split(const matrix &data, uint num_features)
  {
  
   double max_info_gain = -DBL_MAX;
   vector feature_values = {};
   vector left_v={}, right_v={}, y_v={};
   
//---
   
   split_info best_split;
   split_info split;
   
   for (int i=0; i<(int)num_features; i++)
     {
       feature_values = data.Col(i);
       vector possible_thresholds = MatrixExtend::Unique(feature_values);
              
        if (possible_thresholds.Size() <= 1)
           continue; // Skip this feature as it won't provide meaningful splits
             
       //---
             
         for (int j=0; j<(int)possible_thresholds.Size(); j++)
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
                        printf("    split left: [%dx%d] split right: [%dx%d] curr_info_gain: %f max_info_gain: %f",split.dataset_left.Rows(),split.dataset_left.Cols(),split.dataset_right.Rows(),split.dataset_right.Cols(),curr_info_gain,max_info_gain);
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
Node *CDecisionTreeClassifier::build_tree(matrix &data, uint curr_depth=0)
 {
    matrix X;
    vector Y;
    
         
    if (!MatrixExtend::XandYSplitMatrices(data,X,Y)) //Split the input matrix into feature matrix X and target vector Y.    
      {
         #ifdef DEBUG_MODE
            printf("%s Line %d Failed to build a tree Data Empty",__FUNCTION__,__LINE__);
         #endif 
         
         return NULL; //return null pointer
      }
    
    is_fitted = true;
     
    ulong samples = X.Rows(), features = X.Cols(); //Get the number of samples and features in the dataset.
        
    ArrayResize(nodes, nodes.Size()+1); //Append the nodes to memory
    Node *left_child, *right_child;
            
    if (samples >= m_min_samples_split && curr_depth<=m_max_depth)
      {
         split_info best_split = this.get_best_split(data, (uint)features);
         
         #ifdef DEBUG_MODE
             Print(__FUNCTION__," | ",__LINE__,"\nbest_split left: [",best_split.dataset_left.Rows(),"x",best_split.dataset_left.Cols(),"]\nbest_split right: [",best_split.dataset_right.Rows(),"x",best_split.dataset_right.Cols(),"]\nfeature_index: ",best_split.feature_index,"\nInfo gain: ",best_split.info_gain,"\nThreshold: ",best_split.threshold);
         #endif 
                  
         if (best_split.info_gain > 0)
           {
             left_child = this.build_tree(best_split.dataset_left, curr_depth+1);
             right_child = this.build_tree(best_split.dataset_right, curr_depth+1);
                      
             nodes[nodes.Size()-1] = new Node(best_split.feature_index,best_split.threshold,left_child,right_child,best_split.info_gain);  
             return nodes[nodes.Size()-1];
           }
      }      
     
     nodes[nodes.Size()-1] = new Node();
     nodes[nodes.Size()-1].leaf_value = this.calculate_leaf_value(Y);
     
     return nodes[nodes.Size()-1];
 }
//+------------------------------------------------------------------+
//|   returns the element from Y that has the highest count,         |
//|  effectively finding the most common element in the list.        |
//+------------------------------------------------------------------+
double CDecisionTreeClassifier::calculate_leaf_value(vector &Y)
 {   
   vector uniques_count = MatrixExtend::Unique_count(Y);
   vector unique = MatrixExtend::Unique(Y);
   
   return unique[uniques_count.ArgMax()];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDecisionTreeClassifier::make_predictions(const vector &x, const Node &tree)
 {   
   if (!check_is_fitted(__FUNCTION__))
     return 0;
   
   //if (CheckPointer(tree)=POINTER_INVALID)
     
    if (tree.leaf_value != NULL) //This is a leaf_value
      return tree.leaf_value;
    
    #ifdef DEBUG_MODE
      printf("Tree.threshold %f tree.feature_index %d leaf_value %f",tree.threshold,tree.feature_index,tree.leaf_value);
    #endif 
    
    if (tree.feature_index>=x.Size())
      return tree.leaf_value;
          
    double feature_value = x[tree.feature_index];
    double pred = 0;
    
    if (feature_value <= tree.threshold)
      {
       if (CheckPointer(tree.left_child)!=POINTER_INVALID)
          pred = this.make_predictions(x, tree.left_child);  
      }
    else
     {
       if (CheckPointer(tree.right_child)!=POINTER_INVALID)
         pred = this.make_predictions(x, tree.right_child);
     }
     
   return pred;
 }
//+------------------------------------------------------------------+
//|      Commonly used for making predictions in REAL-TIME           |
//+------------------------------------------------------------------+
double CDecisionTreeClassifier::predict(const vector &x)
 {
   if (!check_is_fitted(__FUNCTION__))
     return 0;
          
   return this.make_predictions(x, this.root);
 }
//+------------------------------------------------------------------+
//|   Commonly used for making predictions in TRAIN-TEST             |
//+------------------------------------------------------------------+
vector CDecisionTreeClassifier::predict(const matrix &x)
 {
    vector ret(x.Rows());
 
   if (!check_is_fitted(__FUNCTION__))
     return ret;
        
    for (ulong i=0; i<x.Rows(); i++)
       ret[i] = this.predict(x.Row(i));
       
   return ret;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

class CDecisionTreeRegressor: public CDecisionTreeClassifier
  {
private:
                     double  calculate_leaf_value(vector &Y);
                     split_info  get_best_split(matrix &data, uint num_features);
                     double variance_reduction(vector &parent, vector &l_child, vector &r_child);
                     
                     Node *build_tree(matrix &data, uint curr_depth=0);
public:
                     CDecisionTreeRegressor(uint min_samples_split=2, uint max_depth=2);
                    ~CDecisionTreeRegressor(void);
                    
                     void fit(matrix &x, vector &y);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDecisionTreeRegressor::CDecisionTreeRegressor(uint min_samples_split=2, uint max_depth=2):CDecisionTreeClassifier(min_samples_split, max_depth)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDecisionTreeRegressor::~CDecisionTreeRegressor(void)
 {
 
 }
//+------------------------------------------------------------------+
//|         function to compute variance reduction                   |
//+------------------------------------------------------------------+
double CDecisionTreeRegressor::variance_reduction(vector &parent, vector &l_child, vector &r_child)
 {
    double weight_l = l_child.Size() / (double)parent.Size(),
           weight_r = r_child.Size() / (double)parent.Size();
     
    return parent.Var() - ((weight_l * l_child.Var()) + (weight_r * r_child.Var()));
 }
//+------------------------------------------------------------------+
//|      Return the Node for the best split                          |
//+------------------------------------------------------------------+
split_info CDecisionTreeRegressor::get_best_split(matrix &data, uint num_features)
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
       vector possible_thresholds = MatrixExtend::Unique(feature_values);
                  
         for (uint j=0; j<possible_thresholds.Size(); j++)
            {              
              split = this.split_data(data, i, possible_thresholds[j]);
              
              if (split.dataset_left.Rows()>0 && split.dataset_right.Rows() > 0)
                {
                  y_v = data.Col(data.Cols()-1);
                  right_v = split.dataset_right.Col(split.dataset_right.Cols()-1);
                  left_v = split.dataset_left.Col(split.dataset_left.Cols()-1);
                  
                  double curr_info_gain = this.variance_reduction(y_v, left_v, right_v);
                                    
                  if (curr_info_gain > max_info_gain)
                    {             
                      #ifdef DEBUG_MODE
                        printf(__FUNCTION__," | ",__LINE__,"\nsplit left: [%dx%d] split right: [%dx%d] curr_info_gain: %f max_info_gain: %f",split.dataset_left.Rows(),split.dataset_left.Cols(),split.dataset_right.Rows(),split.dataset_right.Cols(),curr_info_gain,max_info_gain);
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
Node *CDecisionTreeRegressor::build_tree(matrix &data, uint curr_depth=0)
 {
    matrix X;
    vector Y;
      
    if (!MatrixExtend::XandYSplitMatrices(data,X,Y)) //Split the input matrix into feature matrix X and target vector Y.    
      {
         #ifdef DEBUG_MODE 
           printf("%s Line %d Failed to build a tree Data Empty",__FUNCTION__,__LINE__);
         #endif 
         
         return NULL; //Return a NULL pointer
      }
      
    ulong samples = X.Rows(), features = X.Cols(); //Get the number of samples and features in the dataset.
        
    ArrayResize(nodes, nodes.Size()+1); //Append the nodes to memory
    Node *left_child, *right_child;
            
    if (samples >= m_min_samples_split && curr_depth<=m_max_depth)
      {
         split_info best_split = this.get_best_split(data, (uint)features);
         
         #ifdef DEBUG_MODE
           Print(__FUNCTION__," | ",__LINE__,"\nbest_split left: [",best_split.dataset_left.Rows(),"x",best_split.dataset_left.Cols(),"]\nbest_split right: [",best_split.dataset_right.Rows(),"x",best_split.dataset_right.Cols(),"]\nfeature_index: ",best_split.feature_index,"\nInfo gain: ",best_split.info_gain,"\nThreshold: ",best_split.threshold);
         #endif 
                  
         if (best_split.info_gain > 0)
           {
             left_child = this.build_tree(best_split.dataset_left, curr_depth+1);
             right_child = this.build_tree(best_split.dataset_right, curr_depth+1);
                      
             nodes[nodes.Size()-1] = new Node(best_split.feature_index,best_split.threshold,left_child,right_child,best_split.info_gain);  
             return nodes[nodes.Size()-1];
           }
      }      
     
     nodes[nodes.Size()-1] = new Node();
     nodes[nodes.Size()-1].leaf_value = this.calculate_leaf_value(Y);
     
     return nodes[nodes.Size()-1];
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CDecisionTreeRegressor::fit(matrix &x, vector &y)
 {
   matrix data = MatrixExtend::concatenate(x, y, 1);
      
   this.root = this.build_tree(data);
   
   is_fitted = true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CDecisionTreeRegressor::calculate_leaf_value(vector &Y)
 {
   return Y.Mean();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
