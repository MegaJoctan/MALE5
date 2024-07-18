//+------------------------------------------------------------------+
//|                                                initiallizers.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

#include <MALE5\MatrixExtend.mqh>

enum enum_weight_initializers
 {
   WEIGHT_INTIALIZER_XAVIER, //Xavier weights
   WEIGHT_INTIALIZER_HE, //He weights 
   WEIGHT_INTIALIZER_RANDOM //Random weights 
 };

//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
class CWeightsInitializers
  {
protected:

   static double random()  //Generates a random float between 0 (inclusive) and 1 (exclusive)
     {              
       return 0 + double((MathRand() / 32767.0) * (0.9 - 0));
     }
     
   static double uniform(double low, double high)
     {
       return low + (high - low) * random();
     }
     
   static matrix uniform(double low, double high, ulong rows, ulong cols)
    {
      matrix return_matrix(rows, cols);
      for (ulong i=0; i<rows; i++)
        for (ulong j=0; j<cols; j++)
           return_matrix[i][j] = uniform(low, high);
      
      return return_matrix;
    }
   
   static vector uniform(double low, double high, ulong size)
    {
      vector v(size);
      for (ulong i=0; i<size; i++)
        v[i] = uniform(low, high);
      
      return v;
    }
   
public:
                     CWeightsInitializers(void);
                    ~CWeightsInitializers(void);
                    
                    static matrix Xavier(const ulong inputs, const ulong outputs); //Xavier or Glorot initialization
                    static matrix He(const ulong inputs, const ulong outputs);
                    static matrix Random(const ulong inputs, const ulong outputs);
                    static matrix Initializer(const ulong inputs, const ulong outputs, enum_weight_initializers WEIGHT_INIT=WEIGHT_INTIALIZER_XAVIER);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CWeightsInitializers::Xavier(const ulong inputs,const ulong outputs)
 {
   #ifdef RANDOM_STATE
     MathSrand(RANDOM_STATE);
   #endif 
   
   double limit = sqrt(6/(inputs+outputs+DBL_EPSILON));
   return uniform(-limit, limit, inputs, outputs);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CWeightsInitializers::He(const ulong inputs, const ulong outputs)
 {
   #ifdef RANDOM_STATE
     MathSrand(RANDOM_STATE);
   #endif 
   
   double limit = sqrt(2 / (inputs+DBL_EPSILON));
   
   matrix W(inputs, outputs);
   for (ulong i=0; i<outputs; i++)
    {
      vector v = uniform(-limit, limit, inputs);
      W.Col(v, i);
    }
   
   return uniform(-limit, limit, inputs, outputs);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CWeightsInitializers::Random(const ulong inputs,const ulong outputs)
 {
   #ifdef RANDOM_STATE
     MathSrand(RANDOM_STATE);
   #endif 
   
   matrix W(inputs, outputs);
   for (ulong i=0; i<inputs; i++)
     for (ulong j=0; j<outputs; j++)
       W[i][j] = random();
       
   return W;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CWeightsInitializers::Initializer(const ulong inputs,const ulong outputs,enum_weight_initializers WEIGHT_INIT=WEIGHT_INTIALIZER_XAVIER)
 {
   matrix W = {};
   
   switch(WEIGHT_INIT)
     {
      case  WEIGHT_INTIALIZER_HE:
        W = He(inputs, outputs);
        break;
      case  WEIGHT_INTIALIZER_RANDOM:
        W = Random(inputs, outputs);
        break;
      case  WEIGHT_INTIALIZER_XAVIER:
        W = Xavier(inputs, outputs);
        break;
     }
     
   return W;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

