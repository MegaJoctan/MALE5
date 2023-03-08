//+------------------------------------------------------------------+
//|                                                     PCA Test.mq5 |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property description "This is a Test EA file for testing the pca.mqh file for the MALE5 repository located at /Principal Component Analysis(PCA)"

#define DEBUG_MODE

#include <MALE5\Principal Component Analysis(PCA)\pca.mqh>
#include <MALE5\matrix_utils.mqh>
#include <MALE5\MqPlotLib\plots.mqh>

Cpca *pca;
CMatrixutils matrix_utiils;
CPlots *plt;

input criterion ENUM_CRITERION = CRITERION_KAISER;
input int period = 14;
input int bars = 100;
input bool plot_pca = true;

int handles[10];
matrix ind_Matrix(bars,10);
vector buff_v;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
    
//---
   handles[0] = iATR(Symbol(),PERIOD_CURRENT, period);
   handles[1] = iBearsPower(Symbol(), PERIOD_CURRENT, period);
   handles[2] = iMACD(Symbol(),PERIOD_CURRENT,12, 26,9,PRICE_CLOSE);
   handles[3] = iChaikin(Symbol(), PERIOD_CURRENT,12,26,MODE_SMMA,VOLUME_TICK);
   handles[4] = iCCI(Symbol(),PERIOD_CURRENT,period, PRICE_CLOSE);
   handles[5] = iDeMarker(Symbol(),PERIOD_CURRENT,period);
   handles[6] = iForce(Symbol(),PERIOD_CURRENT,period,MODE_EMA,VOLUME_TICK);
   handles[7] = iMomentum(Symbol(),PERIOD_CURRENT,period, PRICE_CLOSE);
   handles[8] = iRSI(Symbol(),PERIOD_CURRENT,period,PRICE_CLOSE);
   handles[9] = iWPR(Symbol(),PERIOD_CURRENT,period);
   
   
   for (int i=0; i<10; i++)
    {
      matrix_utiils.CopyBufferVector(handles[i],0,0,bars,buff_v);
      ind_Matrix.Col(buff_v, i);
    }
   
//---

    Print("Oscillators Correlation Matrix\n",ind_Matrix.CorrCoef(false));   
   
    pca = new Cpca(ind_Matrix);
    matrix pca_matrix = pca.ExtractComponents(ENUM_CRITERION);
    
    //Print("PCA'S\n",pca_matrix);
    
    if (plot_pca)
     {
       plt = new CPlots();

       if (ENUM_CRITERION == CRITERION_SCREE_PLOT) //give room for scree plot
         Sleep(10000);
         
       plt.ScatterCurvePlotsMatrix("pca's ",pca_matrix,"var","PCA");
     }
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   if (CheckPointer(plt)!=ERR_INVALID_POINTER)
     delete(plt);
      
   delete(pca); 
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
  }
//+------------------------------------------------------------------+
