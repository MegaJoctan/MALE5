//+------------------------------------------------------------------+
//|                                                       KMeans.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+

#include <Graphics\Graphic.mqh>
CGraphic graph;
#include <MALE5\matrix_utils.mqh>

//+------------------------------------------------------------------+
enum errors
  {
   KM_ERR001, //clusters not matching in size Error
   KM_ERR002
  };

//+------------------------------------------------------------------+
 
class CKMeans
  {
  CMatrixutils matrix_utils;
  
private:
   ulong             n; //number of samples
   uint              m_clusters;
   ulong             m_cols;
   matrix            InitialCentroids;
   vector<double>    cluster_assign;

protected:
   matrix            Matrix;
   bool              ErrMsg(errors err); 
   bool              ScatterCurvePlots(
                                       string obj_name,
                                       double &x[],
                                       double &y[],
                                       double &curveArr[],
                                       string legend,
                                       string x_axis_label = "x-axis",
                                       string y_axis_label = "y-axis",
                                       color  clr = clrDodgerBlue,
                                       bool   points_fill = true
                                    );

public:
                     CKMeans(const matrix &_Matrix, int clusters=3);
                    ~CKMeans(void);

   void              KMeansClustering(matrix &clustered_matrix, matrix &centroids, int iterations = 1, bool rand_cluster =false);
   void              ElbowMethod(const int initial_k=1, int total_k=10, bool showPlot = true);
   void              FilterZero(vector &vec);
   void              matrixtoArray(matrix &mat, double &Array[]);
  };
//+------------------------------------------------------------------+
CKMeans::CKMeans(const matrix &_Matrix, int clusters=3)
  {
   m_clusters = clusters;
   Matrix.Copy(_Matrix);

   m_cols = Matrix.Cols();
   n = Matrix.Rows(); //number of elements | Matrix Rows
  }
//+------------------------------------------------------------------+
CKMeans::~CKMeans(void)
  {
   ZeroMemory(m_clusters);
   ZeroMemory(InitialCentroids);
   ZeroMemory(cluster_assign);
  }

//+------------------------------------------------------------------+
 
void CKMeans::KMeansClustering(matrix &clustered_matrix, matrix &centroids, int iterations = 1, bool rand_cluster =false)
  {
   InitialCentroids.Resize(m_clusters, m_cols);
   cluster_assign.Resize(n);
   
   clustered_matrix.Resize(m_clusters, m_cols*n);
   clustered_matrix.Fill(NULL);

   vector cluster_comb_v = {};
   matrix cluster_comb_m = {};

   vector rand_v = {};
   ulong rand_ = 0;

   for(ulong i=0; i<m_clusters; i++)
     {
      rand_ = rand_cluster ? (ulong)MathFloor(i*(n/m_clusters)) : i;
      rand_v = Matrix.Row(rand_);

      InitialCentroids.Row(rand_v, i);
     }

//---

   vector v_row;


   matrix rect_distance = {};  //matrix to store rectilinear distances
   rect_distance.Reshape(n, m_clusters);


   vector v_matrix = {}, v_centroid = {};
   double output = 0;

//---

   for(int iter=0; iter<iterations; iter++)
     {

      for(ulong i=0; i<rect_distance.Rows(); i++)
         for(ulong j=0; j<rect_distance.Cols(); j++)
           {
            v_matrix = Matrix.Row(i);
            v_centroid = InitialCentroids.Row(j);

            ZeroMemory(output);

            for(ulong k=0; k<v_matrix.Size(); k++)
               output += MathAbs(v_matrix[k] - v_centroid[k]); //Rectilinear distance

            rect_distance[i][j] = output;
           }

      //---  Assigning the Clusters

      matrix cluster_cent = {}; //cluster centroids
      ulong cluster = 0;

      for(ulong i=0; i<rect_distance.Rows(); i++)
        {
         v_row = rect_distance.Row(i);
         cluster = v_row.ArgMin();

         cluster_assign[i] = (double)cluster;
        }


      vector temp_cluster_assign = cluster_assign;

      //--- Combining the clusters

      for(ulong i=0, index =0; i<cluster_assign.Size(); i++)
        {
         ZeroMemory(cluster_cent);
         bool classified = false;

         for(ulong j=0, count = 0; j<temp_cluster_assign.Size(); j++)
           {

            if(cluster_assign[i] == temp_cluster_assign[j])
              {
               classified = true;

               count++;

               cluster_comb_m.Resize(count, m_cols);

               cluster_comb_m.Row(Matrix.Row(j), count-1);

               cluster_cent.Resize(count, m_cols);

               // New centroids
               cluster_cent.Row(Matrix.Row(j), count-1);

               temp_cluster_assign[j] = -100; //modify the cluster item so it can no longer be found
              }
            else
               continue;
           }

         //---

         if(classified == true)
           {
            // solving for new cluster and updtating the old ones
            
            
             cluster_comb_v = matrix_utils.MatrixToVector(cluster_comb_m);
            

            if(iter == iterations-1)
               clustered_matrix.Row(cluster_comb_v, index); 

            index++;
            //---

            vector x_y_z = {0, 0};
            ZeroMemory(rand_v);

            for(ulong k=0; k<cluster_cent.Cols(); k++)
              {
               x_y_z.Resize(cluster_cent.Cols());
               rand_v = cluster_cent.Col(k);

               x_y_z[k] = rand_v.Mean();
              }

            InitialCentroids.Row(x_y_z, i);

           }

         if(index >= m_clusters)
            break;
        }

     } //end of iterations

   ZeroMemory(centroids);
   centroids.Copy(InitialCentroids);
  }

//+------------------------------------------------------------------+

bool CKMeans::ErrMsg(errors err)
  {
   switch(err)
     {

      case  KM_ERR001:
         printf("%s Clusters not matching in Size ", EnumToString(KM_ERR001));
         break;
      default:
         break;
     }
   return(true);
  }

//+------------------------------------------------------------------+
 
void CKMeans::ElbowMethod(const int initial_k=1, int total_k=10, bool showPlot = true)
  {
   matrix clustered_mat, _centroids = {};

   if(total_k > (int)n)
      total_k = (int)n; //k should always be less than n

   vector centroid_v= {}, x_y_z= {};
   vector short_v = {}; //vector for each point
   vector minus_v = {}; //vector to store the minus operation output

   double wcss = 0;
   double WCSS[];
   ArrayResize(WCSS, total_k);
   double kArray[];
   ArrayResize(kArray, total_k);

   for(int k=initial_k, count_k=0; k<ArraySize(WCSS)+initial_k; k++, count_k++)
     {

      wcss = 0;

      m_clusters = k;

      KMeansClustering(clustered_mat, _centroids);

      for(ulong i=0; i<_centroids.Rows(); i++)
        {
         centroid_v = _centroids.Row(i);

         x_y_z = clustered_mat.Row(i);
         FilterZero(x_y_z);


         for(ulong j=0; j<x_y_z.Size()/m_cols; j++)
           {

            matrix_utils.Copy(x_y_z, short_v, uint(j*m_cols), (uint)m_cols);

            //---                WCSS ( within cluster sum of squared residuals )

            minus_v = (short_v - centroid_v);

            minus_v = MathPow(minus_v, 2);

            wcss += minus_v.Sum();

           }

        }

      WCSS[count_k] = wcss;
      kArray[count_k] = k;
     }

   Print("WCSS");
   ArrayPrint(WCSS);
   Print("kArray");
   ArrayPrint(kArray);

//--- Plotting the Elbow on the graph

   if(showPlot)
     {
      ObjectDelete(0, "elbow");
      ScatterCurvePlots("elbow", kArray, WCSS, WCSS, "Elbow line", "k", "WCSS");
     }
  }

//+------------------------------------------------------------------+
 
void CKMeans::FilterZero(vector &vec)
  {
   vector new_vec = {};

   for(ulong i=0, count =0; i<vec.Size(); i++)
     {
      if(vec[i] != NULL)
        {
         count++;
         new_vec.Resize(count);
         new_vec[count-1] = vec[i];
        }
      else
         continue;
     }

   vec.Copy(new_vec);
  }

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
 
bool CKMeans::ScatterCurvePlots(
   string obj_name,
   double &x[],
   double &y[],
   double &curveArr[],
   string legend,
   string x_axis_label = "x-axis",
   string y_axis_label = "y-axis",
   color  clr = clrDodgerBlue,
   bool   points_fill = true
)
  {

   if(!graph.Create(0, obj_name, 0, 30, 70, 800, 640))
     {
      printf("Failed to Create graphical object on the Main chart Err = %d", GetLastError());
      return(false);
     }

   ChartSetInteger(0, CHART_SHOW, true);

//---

//graph.CurveAdd(x,y,clrBlack,CURVE_POINTS,y_axis_label);
   graph.CurveAdd(x, curveArr, clr, CURVE_POINTS_AND_LINES, legend);

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
 
void CKMeans::matrixtoArray(matrix &mat, double &Array[])
  {
   ArrayFree(Array);
   ArrayResize(Array, int(mat.Rows()*mat.Cols()));

   int index = 0;
   for(ulong i=0; i<mat.Rows(); i++)
      for(ulong j=0; j<mat.Cols(); j++, index++)
        {
         Array[index] = mat[i][j];
        }
  }

//+------------------------------------------------------------------+ 
