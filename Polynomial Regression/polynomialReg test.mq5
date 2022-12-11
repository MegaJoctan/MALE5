//+------------------------------------------------------------------+
//|                                           polynomialReg test.mq5 |
//|                                    Copyright 2022, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"
#property strict
#property script_show_inputs
//+---------------------------------------------------------------------+
//|  For some strange reasons I couldn't get the visuals simultaneously |
//| To show on the chart without them getting into conflict, To see the |
//| graph of BIC or any other visual as described on the Article kindly |
//| remove comment the code on ploting the graph that is on that section|
//| after that comment out the code for the other visuals, Generally    |
//| speaking allow one chart to display per script run.                 |
//+---------------------------------------------------------------------+

#include "Polynomial Regression.mqh";
CPolynomialRegression *pol_reg; 

#include <Graphics\Graphic.mqh>
CGraphic graph;
//---
input bool   ChartShow = false;
input int    bars = 100;
input string X_symbol = "EURUSD";
input string Y_symol = "GBPUSD";
input int    polynomia_degrees = 10;

//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
  
   ObjectsDeleteAll(0,0);
   
   if (!SymbolSelect(X_symbol,true))   { Alert(X_symbol," not found on Market watch Error = ",GetLastError()," Add it to the market watch\nOR check the symbol name spelling on the input X_symbol");  ExpertRemove(); }
   if (!SymbolSelect(Y_symol,true))    { Alert(Y_symol," not found on Market watch Error = ",GetLastError()," Add it to the market watch\nOR check the symbol name spelling on the input Y_symbol");  ExpertRemove(); }
  
   matrix rates(bars, 2);
   vector price_close;

//---

   vector x_v, y_v;

   price_close.CopyRates(X_symbol,PERIOD_H1,COPY_RATES_CLOSE,1,bars); //extracting prices
   
   rates.Col(price_close,0);
   
   x_v.Copy(price_close);

//---
   
   price_close.CopyRates(Y_symol,PERIOD_H1,COPY_RATES_CLOSE,1,bars);
   
   y_v.Copy(price_close);
    
   rates.Col(price_close,1);
   
//---

   MinMaxScaler(x_v); //scalling all the close prices
   MinMaxScaler(y_v); //scalling all the close prices

//---

   pol_reg = new CPolynomialRegression(x_v,y_v,2);
   
   Print("correlation coefficient ",x_v.CorrCoef(y_v));

   string plot_name = "x vs y"; 

/* 
   ObjectDelete(0,plot_name);     
   ScatterPlot(plot_name,x_v,y_v,X_symbol,X_symbol,Y_symol,clrOrange);
*/
   
//--- FINDING BEST MODEL USING BIC
   
   vector bic_; //A vector to store the model BIC values for visualization purposes only
   
   int best_order; //A variable to store the best model order 
   
   pol_reg.BIC(polynomia_degrees,bic_,best_order);
   
   ulong bic_cols = polynomia_degrees-2; //2 is the first in the polynomial order 
   
//--- Plot BIc vs model degrees

   vector x_bic;
   x_bic.Resize(bic_cols);  
   for (ulong i=2,counter =0; i<bic_cols; i++)  {   x_bic[counter] = (double)i;   counter++;   }     
    
/*
   ObjectDelete(0,plot_name);
   plot_name = "curves";
   ScatterCurvePlots(plot_name,x_bic,y_v,bic_,"curves","degree","BIC",clrBlue);
   Sleep(10000);
*/

//--- Plot 

   vector Predictions;
   pol_reg.PolynomialRegressionfx(best_order,Predictions); //Create model with the best order then use it to predict
   
///*
   
   ObjectDelete(0,plot_name); 
   plot_name = "Actual vs predictions";   
   ScatterCurvePlots(plot_name,x_v,y_v,Predictions,string(best_order)+"degree Predictons",X_symbol,Y_symol,clrDeepPink);
   
//*/

   Print("Model Accuracy = ",DoubleToString(pol_reg.r_squared(y_v,Predictions)*100,2),"%");
   
   delete(pol_reg);    
  }
//+------------------------------------------------------------------+
bool ScatterPlot(
                 string obj_name,
                 vector &x,
                 vector &y,
                 string legend,
                 string x_axis_label = "x-axis",
                 string y_axis_label = "y-axis", 
                 color  clr = clrDodgerBlue,
                 bool   points_fill = true                 
                )
 { 
  if (!graph.Create(0,obj_name,0,30,70,600,640))
     {
       printf("Failed to Create graphical object on the Main chart Err = %d",GetLastError());
       return(false);
     }
   
   ChartSetInteger(0,CHART_SHOW,ChartShow);
   
   double x_arr[], y_arr[]; 
   
   pol_reg.vectortoArray(x,x_arr);
   pol_reg.vectortoArray(y,y_arr);
   
   CCurve *curve = graph.CurveAdd(x_arr,y_arr,clr,CURVE_POINTS);
   curve.PointsSize(13);
   curve.PointsFill(points_fill); 
   curve.Name(legend);
   graph.XAxis().Name(x_axis_label);
   graph.XAxis().NameSize(13);
   graph.YAxis().Name(y_axis_label);
   graph.YAxis().NameSize(13);
   graph.FontSet("Lucida Console",13);
   graph.CurvePlotAll();
   graph.Update();
 
   delete(curve);
   
   return(true);
 }
//+------------------------------------------------------------------+

bool ScatterCurvePlots(
                       string obj_name,
                       vector &x,
                       vector &y,
                       matrix &CurveMatrix, 
                       string legend,
                       string x_axis_label = "x-axis",
                       string y_axis_label = "y-axis", 
                       color  clr = clrDodgerBlue,
                       bool   points_fill = true                 
                      )
 {
   
  if (!graph.Create(0,obj_name,0,30,70,600,640))
     {
       printf("Failed to Create graphical object on the Main chart Err = %d",GetLastError());
       return(false);
     }
   
   ChartSetInteger(0,CHART_SHOW,ChartShow);
   
   double x_arr[], y_arr[]; 
   
   pol_reg.vectortoArray(x,x_arr);
   pol_reg.vectortoArray(y,y_arr);
   
//--- additional curves

   vector col_V = {};
   double col_A[]; 
   double CurveMatrix_A[]; //curve matrix array
   
   if (CurveMatrix.Cols() == (CurveMatrix.Rows()*CurveMatrix.Cols()))
     {  
         pol_reg.matrixtoArray(CurveMatrix,CurveMatrix_A);
         graph.CurveAdd(x_arr,y_arr,clrBlack,CURVE_POINTS,y_axis_label);
         graph.CurveAdd(x_arr,CurveMatrix_A,clr,CURVE_POINTS_AND_LINES,legend);
     }
   else
   {
      for (ulong i=0; i<CurveMatrix.Cols(); i++)
        { 
           
          //pol_reg.MatrixPrint(CurveMatrix,1);
          CurveMatrix.Col(col_V,i);
          pol_reg.vectortoArray(col_V,col_A);
          
          graph.CurveAdd(col_A,CURVE_POINTS_AND_LINES," col "+string(i+1)); 
          //break; 
        }
   }
//---
   graph.XAxis().Name(x_axis_label);
   graph.XAxis().NameSize(13);
   graph.YAxis().Name(y_axis_label);
   graph.YAxis().NameSize(13);
   graph.FontSet("Lucida Console",13);
   graph.CurvePlotAll();
   graph.Update();
   

   
   return(true);
 } 
//+------------------------------------------------------------------+


bool ScatterCurvePlots(
                       string obj_name,
                       vector &x,
                       vector &y,
                       vector &curveVector, 
                       string legend,
                       string x_axis_label = "x-axis",
                       string y_axis_label = "y-axis", 
                       color  clr = clrDodgerBlue,
                       bool   points_fill = true                 
                      )
 {
 
  if (!graph.Create(0,obj_name,0,30,70,600,640))
     {
       printf("Failed to Create graphical object on the Main chart Err = %d",GetLastError());
       return(false);
     }
   
   ChartSetInteger(0,CHART_SHOW,ChartShow);
   
   
//--- additional curves

   double x_arr[], y_arr[]; 
   
   pol_reg.vectortoArray(x,x_arr);
   pol_reg.vectortoArray(y,y_arr);

   double curveArray[]; //curve matrix array
   
   pol_reg.vectortoArray(curveVector,curveArray);
   
   graph.CurveAdd(x_arr,y_arr,clrBlack,CURVE_POINTS,y_axis_label);
   graph.CurveAdd(x_arr,curveArray,clr,CURVE_POINTS_AND_LINES,legend);
    
//---

   graph.XAxis().Name(x_axis_label);
   graph.XAxis().NameSize(10);
   graph.YAxis().Name(y_axis_label);
   graph.YAxis().NameSize(10);
   graph.FontSet("Lucida Console",10);
   graph.CurvePlotAll();
   graph.Update();
   
   return(true);
 } 
//+------------------------------------------------------------------+
void MinMaxScaler(vector &v)
 {
   //Normalizing vector using Min max scaler
   
   double min, max, mean;
   min = v.Min();
   max = v.Max();
   mean = v.Mean();
   
   for (int i=0; i<(int)v.Size(); i++)
     v[i] = (v[i] - min) / (max - min);  
   
 }
//+------------------------------------------------------------------+


