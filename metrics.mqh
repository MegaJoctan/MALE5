//+------------------------------------------------------------------+
//|                                                      metrics.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CMetrics
  {
public:
                     CMetrics(void);
                    ~CMetrics(void);
                    
                    double r_squared(vector &A, vector &F); 
                    double adjusted_r(vector &A, vector &F,uint indep_vars=1);
                    double confusion_matrix(vector &A, vector &F,vector &classes, bool plot=true);
                    double RSS(vector &A, vector &F);
                    double MSE(vector &A, vector &F);
                    
  };

//+------------------------------------------------------------------+

CMetrics::CMetrics(void)
  {
  
  }

//+------------------------------------------------------------------+

CMetrics::~CMetrics(void)
 {
 
 }

//+------------------------------------------------------------------+

double CMetrics::r_squared(vector &A,vector &P)
 {
   if (A.Size() != P.Size())
      {
         Print(__FUNCTION__," Vector A and P are not equal in size ");
         return(0);
      }
 
   double tss = 0, //total sum of squares
          rss;     //residual sum of squares
          
   vector c = MathPow(A-A.Mean(),2);
   
   tss = c.Sum();    
   
   c = MathPow(A-P,2);
   
   rss = c.Sum();
   
   return(1-(rss/tss));      
 }
 
//+------------------------------------------------------------------+

double CMetrics::adjusted_r(vector &A,vector &F,uint indep_vars=1)
 {
   if (A.Size() != F.Size())
      {
         Print(__FUNCTION__," Vector A and P are not equal in size ");
         return(0);
      }
      
   double r2 = r_squared(A,F);
   ulong N = F.Size();
   
   return(1-( (1-r2)*(N-1) )/(N - indep_vars -1));
 }
 
//+------------------------------------------------------------------+

double CMetrics::confusion_matrix(vector &A,vector &F,vector &classes,bool plot=true)
 {    
    matrix conf_m(classes.Size(),classes.Size());
    conf_m.Fill(0);
    vector diag(classes.Size()); 
    
    vector row(classes.Size());
    vector counter(classes.Size());
    
      for (ulong c=0; c<classes.Size(); c++)
       { 
          for (ulong i=0; i<A.Size(); i++) 
              {   
                  if (classes[c] == A[i])
                    {
                     counter[c] = counter[c]+=1;
                     
                     Print("c ",c," i ",i," classes size ",classes.Size()," A size ",A.Size()," counter ",counter[c]);
                     row[c] = counter[c];
                    }
                       
                 if (i > 10 ) break;
              }
          
          counter.Fill(0);
          
          Print("Row ",c," ",row);
       }
     //Print("Diagonal ",diag);
     Print("confusion_matrix\n",conf_m);
     
    return (0);
 }

//+------------------------------------------------------------------+

double CMetrics::RSS(vector &A,vector &F)
 {
   vector c = A-F;
   c = MathPow(c,2);
   
   return (c.Sum()); 
 }

//+------------------------------------------------------------------+

double CMetrics::MSE(vector &A,vector &F)
 {
   vector c = A - F;
   c = MathPow(c,2);
   
   return(c.Sum()/c.Size()); 
 }

//+------------------------------------------------------------------+
