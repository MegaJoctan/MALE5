//+------------------------------------------------------------------+
//|                                        KNN_nearest_neighbors.mqh |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+

bool isdebug = true;

//+------------------------------------------------------------------+ 

class CKNNNearestNeighbors
  {
private:
   uint              k;
   matrix<double>    Matrix;
   ulong             m_rows, m_cols;
   vector            m_target;
   vector            m_classesVector;
   matrix            m_classesMatrix;

   double            Euclidean_distance(const vector &v1, const vector &v2);
   vector            ClassVector(); //global vector of target classes
   void              MatrixRemoveRow(matrix &mat, ulong row);
   void              VectorRemoveIndex(vector &v, ulong index);
   double            Mse(vector &A, vector &P);
public:
                     CKNNNearestNeighbors(matrix<double> &Matrix_, uint k_);
                     CKNNNearestNeighbors(matrix<double> &Matrix_);
                    ~CKNNNearestNeighbors(void);

   int               KNNAlgorithm(vector &vector_);
   vector            CrossValidation_LOOCV(uint initial_k = 0, uint final_k=1); //Leave One out Cross Validation (LOOCV)
   matrix            ConfusionMatrix(vector &A,vector &P);
   float             TrainTest(double train_size = 0.7); //returns accuracy of the tested dataset
  };

//+------------------------------------------------------------------+

CKNNNearestNeighbors:: CKNNNearestNeighbors(matrix<double> &Matrix_, uint k_)
  {
   k = k_;
   
   if(k %2 ==0)
      {
         k = k+1;
         if (isdebug)
            printf("K = %d is an even number, It will be added by One so it becomes an odd Number %d", k_, k);
      }

   Matrix.Copy(Matrix_);

   m_rows = Matrix.Rows();
   m_cols = Matrix.Cols();

   m_target = Matrix.Col(m_cols-1);
   m_classesVector = ClassVector();
   
   if (isdebug)
      Print("classes vector | Neighbors ", m_classesVector);
  }

//+------------------------------------------------------------------+

CKNNNearestNeighbors::CKNNNearestNeighbors(matrix<double> &Matrix_)
  {
   Matrix.Copy(Matrix_);

   k = (int)round(MathSqrt(Matrix.Rows()));
   k = k%2 ==0 ? k+1 : k; //make sure the value of k ia an odd number

   m_rows = Matrix.Rows();
   m_cols = Matrix.Cols();

   m_target = Matrix.Col(m_cols-1);
   m_classesVector = ClassVector();
   Print("classes vector | Neighbors ", m_classesVector);
  }

//+------------------------------------------------------------------+

CKNNNearestNeighbors::~CKNNNearestNeighbors(void)
  {
   ZeroMemory(k);
   ZeroMemory(m_classesVector);
   ZeroMemory(m_classesMatrix);
  }

//+------------------------------------------------------------------+ 

int CKNNNearestNeighbors::KNNAlgorithm(vector &vector_)
  {
   vector vector_2 = {};
   vector euc_dist;
   euc_dist.Resize(m_rows);

   //matrix temp_matrix = Matrix;
   //temp_matrix.Resize(Matrix.Rows(), Matrix.Cols()-1); //remove the last column of independent variables
   
   for(ulong i=0; i<m_rows; i++)
     {
      vector_2 = Matrix.Row(i);
      vector_2.Resize(m_cols-1);
       
      euc_dist[i] = NormalizeDouble(Euclidean_distance(vector_, vector_2), 5);
     }

//---   

   if(isdebug)
     {
      matrix dbgMatrix = Matrix; //temporary debug matrix
      dbgMatrix.Resize(dbgMatrix.Rows(), dbgMatrix.Cols()+1);
      dbgMatrix.Col(euc_dist, dbgMatrix.Cols()-1);

      //Print("Matrix w Euclidean Distance\n",dbgMatrix);

      ZeroMemory(dbgMatrix);
     }

//---
   
   uint size = (uint)m_target.Size();

   double tarArr[];
   ArrayResize(tarArr, size);
   double eucArray[];
   ArrayResize(eucArray, size);

   for(ulong i=0; i<size; i++)  //convert the vectors to array
     {      
      tarArr[i] = m_target[i];
      eucArray[i] = euc_dist[i];
     }

   double track[], NN[];
   ArrayCopy(track, tarArr);


   int max; 
   
   for(int i=0; i<(int)m_target.Size(); i++)
     {
      if(ArraySize(track) > (int)k)
        {
         max = ArrayMaximum(eucArray);
         ArrayRemove(eucArray, max, 1);
         ArrayRemove(track, max, 1);
        }
     }
   ArrayCopy(NN, eucArray);
/*
   Print("NN ");
   ArrayPrint(NN);
   Print("Track ");
   ArrayPrint(track);
*/
//--- Voting process

   vector votes(m_classesVector.Size());

   for(ulong i=0; i<votes.Size(); i++)
     {
      int count = 0;
      for(ulong j=0; j<track.Size(); j++)
        {
         if(m_classesVector[i] == track[j])
            count++;
        }

      votes[i] = (double)count;

      if(votes.Sum() == k)  //all members have voted
         break;
     }

   if(isdebug)
      Print(vector_, " belongs to class ", (int)m_classesVector[votes.ArgMax()]);
      
     return((int)m_classesVector[votes.ArgMax()]);
  }

//+------------------------------------------------------------------+
 
double CKNNNearestNeighbors:: Euclidean_distance(const vector &v1, const vector &v2)
  {
   double dist = 0;

   if(v1.Size() != v2.Size())
      Print(__FUNCTION__, " v1 and v2 not matching in size");
   else
     {
      double c = 0;
      for(ulong i=0; i<v1.Size(); i++)
         c += MathPow(v1[i] - v2[i], 2);

      dist = MathSqrt(c);
     }

   return(dist);
  }

//+------------------------------------------------------------------+
 
vector CKNNNearestNeighbors::ClassVector()
  {
   vector t_vectors = Matrix.Col(m_cols-1); //target variables are found on the last column in the matrix
   vector temp_t = t_vectors, v = {t_vectors[0]};

   for(ulong i=0, count =1; i<m_rows; i++)  //counting the different neighbors
     {
      for(ulong j=0; j<m_rows; j++)
        {
         if(t_vectors[i] == temp_t[j] && temp_t[j] != -1000)
           {
            bool count_ready = false;

            for(ulong n=0; n<v.Size(); n++)
               if(t_vectors[i] == v[n])
                  count_ready = true;

            if(!count_ready)
              {
               count++;
               v.Resize(count);

               v[count-1] = t_vectors[i];

               temp_t[j] = -1000; //modify so that it can no more be counted
              }
            else
               break;
            //Print("t vectors vector ",t_vectors);
           }
         else
            continue;
        }
     }

   return(v);
  }

//+------------------------------------------------------------------+

vector CKNNNearestNeighbors::CrossValidation_LOOCV(uint initial_k = 0, uint final_k=1)
  {
 
    uint iterations = final_k-initial_k;
    
     vector cv(iterations); 
        
      ulong N = m_rows;
      
      matrix OG_Matrix = Matrix; //The original matrix
      ulong OG_rows = m_rows; //the primarly value of rows
      vector OG_target = m_target;
      
      ulong size = N-1;
      
      m_rows = m_rows-1;  //leavo one row out
      
      for (uint z = initial_k; z<final_k; z++)
        { 
          if (iterations>1) k = z;
               
          double sum_mse = 0;
          
          for (ulong i=0; i<size; i++)
            { 
               MatrixRemoveRow(Matrix,i);
               m_target.Resize(m_rows);
               
               vector P(1), A = { OG_target[i] }, v;
               
                    {
                        v = OG_Matrix.Row(i);
                        v.Resize(m_cols-1);
                        
                        P[0] = KNNAlgorithm(v);
                    }
               
               
               if (isdebug)
                   Print("\n Actual ",A," Predicted ",P," ",i," MSE = ",Mse(A,P),"\n");
                
               sum_mse += Mse(A,P);
               
               Matrix.Copy(OG_Matrix);
            }
         
         cv[z] = (float)sum_mse/size;
       } 
      
      Matrix.Copy(OG_Matrix);
      m_rows = OG_rows;
      m_target = OG_target;
      
      return (cv);
  }

//+------------------------------------------------------------------+


void CKNNNearestNeighbors::MatrixRemoveRow(matrix &mat,ulong row)
 {
    matrix new_matrix(mat.Rows()-1,mat.Cols()); //Remove the one column
    
     for (ulong j=0; j<mat.Cols(); j++)
      for (ulong i=0, new_rows=0; i<mat.Rows(); i++)
           {
               if (i == row) continue; 
               else  
                 {
                   new_matrix[new_rows][j] = mat[i][j];
                 
                   new_rows++;
                 }
           }
           
    mat.Copy(new_matrix);
 }
//+------------------------------------------------------------------+

void CKNNNearestNeighbors::VectorRemoveIndex(vector &v, ulong index)
 {
   vector new_v(v.Size()-1);
   
   for (ulong i=0, count = 0; i<v.Size(); i++)
      if (i == index)
        {
          new_v[count] = new_v[i];
          count++;
        }
 }

//+------------------------------------------------------------------+

double CKNNNearestNeighbors::Mse(vector &A,vector &P)
 {
   double err = 0;
   vector c;
   
    if (A.Size() != P.Size()) 
      Print(__FUNCTION__," Err, A and P vectors not the same in size");
    else
      {
         ulong size = A.Size();
         c.Resize(size);
         
         c = MathPow(A - P,2); 
         
         err = (c.Sum()) / size;          
      }
    return (err);
 }

//+------------------------------------------------------------------+
matrix CKNNNearestNeighbors::ConfusionMatrix(vector &A,vector &P)
 {
   ulong size = m_classesVector.Size();
   matrix mat_(size,size);
   
   if (A.Size() != P.Size()) 
      Print("Cant create confusion matrix | A and P not having the same size ");
   else
     {
      
         int tn = 0,fn =0,fp =0, tp=0;
         for (ulong i = 0; i<A.Size(); i++)
            {              
               if (A[i]== P[i] && P[i]==m_classesVector[0])
                  tp++; 
               if (A[i]== P[i] && P[i]==m_classesVector[1])
                  tn++;
               if (P[i]==m_classesVector[0] && A[i]==m_classesVector[1])
                  fp++;
               if (P[i]==m_classesVector[1] && A[i]==m_classesVector[0])
                  fn++;
            }
            
       mat_[0][0] = tn; mat_[0][1] = fp;
       mat_[1][0] = fn; mat_[1][1] = tp;

    }
     
   return(mat_);
    
 }
//+------------------------------------------------------------------+
float CKNNNearestNeighbors::TrainTest(double train_size=0.700000)
 {
//--- Split the matrix
   
   matrix default_Matrix = Matrix; 
   
   int train = (int)MathCeil(m_rows*train_size),
       test  = (int)MathFloor(m_rows*(1-train_size));
   
   if (isdebug) printf("Train %d test %d",train,test);

   matrix TrainMatrix(train,m_cols), TestMatrix(test,m_cols);
   int train_index = 0, test_index =0;

//---
   
   for (ulong r=0; r<Matrix.Rows(); r++)
      {
         if ((int)r < train)
           {
             TrainMatrix.Row(Matrix.Row(r),train_index);
             train_index++;
           }
         else
           {
             TestMatrix.Row(Matrix.Row(r),test_index);
             test_index++;
           }     
      }

   if (isdebug) 
    Print("TrainMatrix\n",TrainMatrix,"\nTestMatrix\n",TestMatrix);
   
//--- Training the Algorithm
   
   Matrix.Copy(TrainMatrix); //That's it ???
   
//--- Testing the Algorithm
   
   vector TestPred(TestMatrix.Rows());
   vector TargetPred = TestMatrix.Col(m_cols-1);
   vector v_in = {};
   
   for (ulong i=0; i<TestMatrix.Rows(); i++)
     {
        v_in = TestMatrix.Row(i);
        v_in.Resize(v_in.Size()-1); //Remove independent variable
        
        TestPred[i] = KNNAlgorithm(v_in);        
     }
   
   matrix cf_m = ConfusionMatrix(TargetPred,TestPred);
   vector diag = cf_m.Diag();
   float acc = (float)(diag.Sum()/cf_m.Sum())*100;
   
   Print("Confusion Matrix\n",cf_m,"\nAccuracy ------> ",acc,"%");
   
   return(acc);      
 }
//+------------------------------------------------------------------+