//+------------------------------------------------------------------+
//|                                                      Tensors.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

#define TENSOR_COLS 2

class CMatrix
  {
   public:
         matrix Matrix;
  };
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

class CTensors
  {
CMatrix* matrices[]; 

public:
                     CTensors(uint DIM); //For one dimension tensor
                    ~CTensors(void);
                    
                    uint   TENSOR_DIMENSION;
                    
                    template<typename T>
                    bool   Add(matrix<T> &mat_ , ulong POS);
                    template<typename T>
                    bool   Append(vector<T> &v, ulong POS);
                    
                    void   Print_();
                    matrix Get(ulong POS);
                    template<typename T>
                    void   Fill(T value);
                    void   MemoryClear();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensors::CTensors(uint DIM)
 {
   TENSOR_DIMENSION = DIM;
   
   ArrayResize(matrices, TENSOR_DIMENSION);
   
   for (uint i=0; i<TENSOR_DIMENSION; i++)
       matrices[i] = new CMatrix;
     
   
   for (uint i=0; i<TENSOR_DIMENSION; i++)
     if (CheckPointer(matrices[i]) == POINTER_INVALID)
       {
         printf("Can't create a tensor, Invalid pointer Err %d ",GetLastError());
         return;
       }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensors::~CTensors(void)
 {
   for (uint i=0; i<TENSOR_DIMENSION; i++)
     if (CheckPointer(matrices[i]) != POINTER_INVALID)
       delete matrices[i];

   ArrayFree(matrices);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
bool  CTensors::Add(matrix<T> &mat_ , ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       return (false);
     }
     
    this.matrices[POS].Matrix = mat_;
   
   return (true);
 }
//+------------------------------------------------------------------+
//|      Appending rows to the Get Matrix at POS index               |
//+------------------------------------------------------------------+
template<typename T>
bool CTensors::Append(vector<T> &v, ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       return (false);
     }
     
//---

   matrix<T> mat = this.matrices[POS].Matrix;
   
   mat.Resize(mat.Rows()+1, mat.Cols());
   mat.Row(v, mat.Rows()-1);
   
   Add(mat, POS);
   
  return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensors::Print_(void)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
     Print("TENSOR INDEX <<",i,">>\n",this.matrices[i].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix CTensors::Get(ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       matrix mat = {};
       return (mat);
     }
     
   return (this.matrices[POS].Matrix); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
template<typename T>
void CTensors::Fill(T value)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
     this.matrices[i].Matrix.Fill(value);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensors::MemoryClear(void)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
    {
      this.matrices[i].Matrix.Resize(1,0);
      ZeroMemory(this.matrices[i].Matrix);
    }
 }

//+------------------------------------------------------------------+
//|                                                                  |
//|   Tensorflows for Vector type of data                            |
//|                                                                  |
//+------------------------------------------------------------------+

class CVectors
  {
   public:
          vector Vector;
  };

//---

class CTensorsVectors
  {
CVectors             *vectors[];

private:
   uint TENSOR_DIMENSION;

public:
                     CTensorsVectors(uint DIM);
                    ~CTensorsVectors(void);
                    
                     bool Add(vector &v, ulong POS);
                     void Print_(void);
                     vector Get(ulong POS);
                     void Fill(double value);
                     void MemoryClear();
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensorsVectors::CTensorsVectors(uint DIM)
 {
   TENSOR_DIMENSION = DIM;
   
   ArrayResize(vectors, TENSOR_DIMENSION);
   
   
   for (uint i=0; i<TENSOR_DIMENSION; i++)
       vectors[i] = new CVectors;
     
   
   for (uint i=0; i<TENSOR_DIMENSION; i++)
     if (CheckPointer(vectors[i]) == POINTER_INVALID)
       {
         printf("Can't create a tensor, Invalid pointer Err %d ",GetLastError());
         return;
       }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CTensorsVectors::~CTensorsVectors(void)
 {
   for (uint i=0; i<TENSOR_DIMENSION; i++)
     if (CheckPointer(vectors[i]) != POINTER_INVALID)
       delete vectors[i];

   ArrayFree(vectors);
 } 
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool  CTensorsVectors::Add(vector &v, ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       return (false);
     }
     
    this.vectors[POS].Vector = v;
   
   return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensorsVectors::Print_(void)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
     Print("TENSOR INDEX <<",i,">>\n",this.vectors[i].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector CTensorsVectors::Get(ulong POS)
 {
   if (POS > TENSOR_DIMENSION) 
     {
       Print(__FUNCTION__," Index Error POS =",POS," greater than TENSOR_DIM ",TENSOR_DIMENSION);
       
       vector v = {};
       return (v);
     }
     
   return (this.vectors[POS].Vector); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensorsVectors::Fill(double value)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
     this.vectors[i].Vector.Fill(value);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTensorsVectors::MemoryClear(void)
 {
   for (ulong i=0; i<TENSOR_DIMENSION; i++)
    {
      this.vectors[i].Vector.Resize(1,0);
      ZeroMemory(this.vectors[i].Vector);
    }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
