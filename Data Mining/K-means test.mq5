//+------------------------------------------------------------------+
//|                                                 K-means test.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property strict
#property script_show_inputs

//+------------------------------------------------------------------+

#include "KMeans.mqh";
CKMeans *clustering;

bool ChartShow  = true;

enum plot_enum
  { CLUSTER_PLOT, ELBOW_PLOT };

input plot_enum PlotOnChart = CLUSTER_PLOT;
input int input_clusters = 3;
input int MATRIXDIMENSION = 1;

input group "ELBOW METHOD";

input int init_clusters = 1;
input int k_clusters = 10;

input group "BARS";

input int bars = 20;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {

   matrix DMatrix = {};
   DMatrix.Resize(bars, MATRIXDIMENSION); //columns determines the dimension of the dataset 1D won't be visualized properly

   vector column_v = {};

   ulong start = 0;
   for(ulong i=0; i<(ulong)MATRIXDIMENSION; i++)
     {
      column_v.CopyRates(Symbol(), PERIOD_CURRENT, COPY_RATES_CLOSE, start, bars);
      DMatrix.Col(column_v, i);

      start += bars;
     }

//---

   MeanNormalization(DMatrix);

   matrix clusterd_mat= {}, centroids_mat = {};

   clustering = new CKMeans(DMatrix, input_clusters);
   clustering.KMeansClustering(clusterd_mat, centroids_mat, k_clusters, false);

   Print("clustered matrix\n", clusterd_mat, "\ncentroids_mat\n", centroids_mat);

   bool elbow_show = false;

   if(PlotOnChart == CLUSTER_PLOT)
     {
      ObjectDelete(0, "graph");
      ObjectDelete(0, "elbow");
      ScatterPlotsMatrix("graph", clusterd_mat, "cluster 1");
     }
   else
      elbow_show = true;

   Sleep(100);
   clustering.ElbowMethod(init_clusters, k_clusters, elbow_show);


   delete(clustering);
  }

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool ScatterPlotsMatrix(
   string obj_name,
   matrix &_matrix,
   string legend,
   string x_axis_label = "x-axis",
   string y_axis_label = "y-axis",
   color  clr = clrDodgerBlue,
   bool   points_fill = true
)
  {

   if(!graph.Create(0, obj_name, 0, 30, 70, 600, 640))
     {
      printf("Failed to Create graphical object on the Main chart Err = %d", GetLastError());
      return(false);
     }

   ChartSetInteger(0, CHART_SHOW, ChartShow);

   double x_arr[], y_arr[];
   vector x = {}, y = {};

   y = _matrix.Row(0);
   x = _matrix.Row(1);

   clustering.FilterZero(x);
   clustering.FilterZero(y);

   vectortoArray(x, x_arr);
   vectortoArray(y, y_arr);

//--- additional curves

//graph.CurveAdd(y_arr,y_arr,clrBlack,CURVE_POINTS,y_axis_label);


   for(ulong i=0; i<_matrix.Rows(); i++)
     {
      x = _matrix.Row(i);

      clustering.FilterZero(x);
      vectortoArray(x, x_arr);

      graph.CurveAdd(x_arr, CURVE_POINTS, " cluster "+string(i+1));
     }

//---

   graph.XAxis().Name(x_axis_label);
   graph.XAxis().NameSize(13);
   graph.YAxis().Name(y_axis_label);
   graph.YAxis().NameSize(13);
   graph.FontSet("Lucida Console", 13);
   graph.CurvePlotAll();
   graph.Update();

   return(true);
  }

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void vectortoArray(vector &v, double &Arr[])
  {
   ArrayResize(Arr, (int)v.Size());

   for(int i=0; i<(int)v.Size(); i++)
     { Arr[i] = v[i];  }

  }

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MeanNormalization(matrix &mat)
  {

   vector v = {};

   for(ulong i=0; i<mat.Cols(); i++)
     {
      v = mat.Col(i);
      MeanNormalization(v);
      mat.Col(v, i);
     }
  }

//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MeanNormalization(vector &v)
  {
   double mean = v.Mean(),
          max = v.Max(),
          min = v.Min();

   for(ulong i=0; i<v.Size(); i++)
      v[i] = (v[i] - mean) / (max - min);

  }

//+------------------------------------------------------------------+
