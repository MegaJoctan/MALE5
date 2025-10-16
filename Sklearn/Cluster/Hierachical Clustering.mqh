//+------------------------------------------------------------------+
//|                                       Hierachical Clustering.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include "Base.mqh"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+

enum linkage_enum 
 {
   LINKAGE_SINGLE,
   LINKAGE_COMPLETE,
   LINKAGE_AVG
 };
 
class CAgglomerativeClustering
  {
protected:

   linkage_enum linkage;
   vector       labels;
   matrix       clusters_keys;
   
   matrix       calc_distance_matrix(matrix &x, vector &cluster_members);   
   
public:
                     CAgglomerativeClustering(linkage_enum linkage_type=LINKAGE_SINGLE);
                    ~CAgglomerativeClustering(void);
                    
                    void fit(matrix &x);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAgglomerativeClustering::CAgglomerativeClustering(linkage_enum linkage_type=LINKAGE_SINGLE)
 :linkage(linkage_type)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CAgglomerativeClustering::~CAgglomerativeClustering(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CAgglomerativeClustering::calc_distance_matrix(matrix &x, vector &cluster_members)
 {
   clusters_keys.Init(1, x.Cols()); //initializes the clusters_keys such that each data point is initially in its own cluster
   for (ulong i=0; i<x.Cols(); i++)    clusters_keys.Col(clusters_keys.Col(i).Fill(i), i); //Filll the initial clusters_keys matrix with their columns incremental values
   
   matrix distance(x.Rows(), x.Rows());
   distance.Fill(0.0);
   
   vector v1, v2;
   
   vector ith_element, jth_element;
   for (ulong i=0; i<distance.Cols(); i++)
    {
     ith_element = cluster_members[(int)clusters_keys[i]];
     for (ulong j=0; j<distance.Cols(); j++)
        {
          jth_element = cluster_members[(int)clusters_keys[j]];
          
          v1 = x.Col(i); v2 = x.Col(j);
          distance[i][j] = Base::norm(v1, v2);
        }
    }
   
   return distance;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CAgglomerativeClustering::fit(matrix &x)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
