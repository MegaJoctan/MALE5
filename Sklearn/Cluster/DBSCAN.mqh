//+------------------------------------------------------------------+
//|                                                       DBSCAN.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\MatrixExtend.mqh>
#include <MALE5\linalg.mqh>

//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CDBSCAN
  {
protected:
   double m_epsilon;
   uint m_minsamples;
   vector labels;
   
   vector get_neighbors(matrix &x, ulong point_index);
   bool expand_cluster(matrix &x, ulong point_index, ulong cluster_id, const vector &neighbors, const vector &cluster_labels);
   
public:
                     CDBSCAN(double epsilon, uint min_samples);
                    ~CDBSCAN(void);
                    
                    vector fit_predict(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDBSCAN::CDBSCAN(double epsilon, uint min_samples):
m_epsilon(epsilon),
m_minsamples(min_samples)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CDBSCAN::~CDBSCAN(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDBSCAN::get_neighbors(matrix &x, ulong point_index)
 {
   vector neighbors = {};
   
   for (ulong i=0, count=0; i<x.Rows(); i++)
    {
      if (LinAlg::norm(x.Row(point_index), x.Row(i)) < this.m_epsilon && i != point_index)
       { 
         count++;
         neighbors.Resize(count);
         neighbors[count-1] = (int)i; //Append the neighbor
       }
    }
   
   return neighbors;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CDBSCAN::expand_cluster(matrix &x, ulong point_index, ulong cluster_id, const vector &neighbors, const vector &cluster_labels)
 {
 
   this.labels[point_index] = (int)cluster_id;
   vector points_to_explore = {(int)point_index};
   
   Print("points to explore in a cluster: ",points_to_explore);
   
   while (points_to_explore.Size()>0)
     {
       vector current_point = {points_to_explore[0]};
       points_to_explore.Resize(0);
       
       Print("current point: ", current_point);
       
       vector current_neighbors = this.get_neighbors(x, (int)current_point[0]);
       
       if (current_neighbors.Size() >= this.m_minsamples)
         for (int neighbor_index=0; neighbor_index<(int)current_neighbors.Size(); neighbor_index++)
           {
             if (this.labels[neighbor_index] == 0 || this.labels[neighbor_index] == -1)
               {
                  if (this.labels[neighbor_index] == 0)
                    {
                      points_to_explore.Resize(1);
                      points_to_explore[0] = neighbor_index;
                      
                      Print("points to explore: ",points_to_explore);
                    }
                 this.labels[neighbor_index] = (int)cluster_id;
               }  
           }
     }
   
   return true; 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CDBSCAN::fit_predict(matrix &x)
 {
   this.labels.Resize(x.Rows()); labels.Fill(0);
   ulong cluster_id = 0; 
   
   for (ulong i=0; i<x.Rows(); i++) 
     {
       if (this.labels[i] != 0)
         continue;
        
        vector neighbors = get_neighbors(x, i);
        
        
        if (neighbors.Size() < this.m_minsamples)
          this.labels[i] = -1; //Mark as noise
        else
          {
           cluster_id++;
           Print("Expand cluster: ",i);
           expand_cluster(x, i, cluster_id, neighbors, this.labels);
          }  
     }
   return this.labels;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

