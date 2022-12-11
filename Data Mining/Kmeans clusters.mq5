//+------------------------------------------------------------------+
//|                                              Kmeans clusters.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


#include <Math\Alglib\alglib.mqh>
#include <Math\Stat\Normal.mqh>

//#include "KMeans.mqh";
//CKMeans  *k_means;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//---
   double Array_A_1[5] = {3, -2, 0, -1, 2};
   double Array_A_2[5] = {10, 10, 20, 10 , 10};
   double Array_B_1[5] = {30, 40, 60, 60, 65};
   double Array_B_2[5] = {-0.5, -1, -0.5, 2, 4};
   
   int k = 2; 
   int vars = 4;
   int npoints = 5;
   int info;
   int membership[];
 
   CMatrixDouble xy(npoints,vars);
   CMatrixDouble clusters;
   
   for (int i=0; i< npoints; i++){
      xy[i].Set(0,Array_A_1[i]);
      xy[i].Set(1,Array_A_2[i]);
      xy[i].Set(2,Array_B_1[i]);
      xy[i].Set(3,Array_B_2[i]);
   }

   CKMeans::KMeansGenerate(xy, npoints, vars,  k, 3, info, clusters ,membership);
   
   Print("Info ",info,"\n", 
         "npoints ",npoints,"\n",
         "vars ",vars,"\n",
         "k ",k,"\n"
         );

//--- 
   
   if (info == 1)
      {
         Print("Clusters "); ArrayPrint(membership);
      }  
  
 
   for (int j=0; j< npoints; j++)
     {
       for (int i=0; i< vars; i++)
         PrintFormat("%f ------> K %f , Dimension %f, M_cluster %f ",xy[i][j],j,i,clusters[i][j]);
     } 

//--- Let's try finding the clusters using my library first

   Print("\n------> mylib <-------\n");

/*
   matrix<double> Matrix;
   Matrix.Resize(npoints,vars);
   
   vector A_1 = {3, -2, 0, -1, 2};
   Matrix.Col(A_1,0);
   vector A_2 = {10, 10, 20, 10 , 10};
   Matrix.Col(A_2,1);
   vector B_1 = {30, 40, 60, 60, 65};
   Matrix.Col(B_1,2);
   vector B_2 = {-0.5, -1, -0.5, 2, 4};
   Matrix.Col(B_2,3);
   
   Matrix = Matrix.Transpose(); 
   
   /* we transpose the matrix to make it human readable In other words to convert it from the 
   array format as that was passed from the vectors, This might not be the case always    
   */
/* 
   Print("Matrix\n",Matrix);
   
   k_means = new CKMeans(Matrix,k);   
   
   matrix<double> clusters_mat, centroids_m; //centroids matrix can become useful when you want to plot the results
   k_means.KMeansClustering(clusters_mat,centroids_m);
   
   Print("clusters matrix\n",clusters_mat);
   
   clusters_mat = clusters_mat.Transpose(); //columns into rows and viceversa, This is crucial because the obtained clusters were placed along the rows
   WriteCsv("Clusters Matrix.csv",clusters_mat); //Save the clusters matrix to a csv file for further aanlysis or external usage

*/  
   
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+

void OnDeinit(const int reason)
  {
//---
      //if (reason == REASON_RECOMPILE)
            //delete(k_means);
      
  }
  
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+

void OnTick()
  {
//---
   
  }
  
//+------------------------------------------------------------------+

void WriteCsv(string name, matrix &Matrix)
 {
   FileDelete(name);
    int handle = FileOpen(name,FILE_WRITE|FILE_CSV|FILE_ANSI,",",CP_UTF8);
     
    ResetLastError();
     
     if (handle == INVALID_HANDLE)
       printf("Invalid %s handle Error %d ",name,GetLastError());
     else
       {
        string concstring;
        vector row;
        FileSeek(handle,0,SEEK_SET);
        
         for (ulong i=0; i<Matrix.Rows(); i++)
            {
               ZeroMemory(concstring);
               
               row = Matrix.Row(i);
                for (ulong j=0, cols =1; j<row.Size(); j++, cols++)
                  {
                    //non zero at the end filter
                    double rest_sum = 0;
                    if (row[j] == 0) {
                        for (ulong k=j; k<row.Size(); k++) rest_sum += row[k];
                        
                        if (rest_sum == 0) break;
                     }
                    //---
                    
                     concstring += (string)ND(row[j]) + (cols == Matrix.Cols() ? "" : ",");
                  }
                  
               //Print(concstring);
               
               FileSeek(handle,0,SEEK_END);
               FileWrite(handle,concstring);
            }
       }  
     FileClose(handle);
 }
 
//+------------------------------------------------------------------+

double ND(double value, int digits=5)
 {
   return(NormalizeDouble(value,digits));
 }

//+------------------------------------------------------------------+