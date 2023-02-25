//+------------------------------------------------------------------+
//|                                                        plots.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"
//+------------------------------------------------------------------+
//| defines                                                          |
//+------------------------------------------------------------------+
#include <Graphics\Graphic.mqh>

class CPlots
  {  
protected:
   CGraphic graph;
   
   long m_chart_id;
   int m_subwin;
   int m_x1, m_x2;
   int m_y1, m_y2;
   bool m_chart_show;
   
   string m_plot_names[];

public:
         CPlots(void);
        ~CPlots(void);
         
         
         void PlotConfigs(long chart_id=0, int sub_win=0 ,int x1=30, int y1=70, int x2=550, int y2=350, bool chart_show=true);
         bool ScatterCurvePlots(string plot_name,double &x[], double &y[], string legend, string x_axis_label = "x-axis", string y_axis_label = "y-axis",color  clr = clrDodgerBlue, bool   points_fill = true);
  };
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPlots::CPlots(void)
 {
   PlotConfigs();
   ChartRedraw(m_chart_id);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPlots::~CPlots(void)
 {
   for (int i=0; i<ArraySize(m_plot_names); i++)
       ObjectDelete(m_chart_id,m_plot_names[i]);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPlots::ScatterCurvePlots(
                                 string plot_name,
                                 double &x[],
                                 double &y[],
                                 string legend,
                                 string x_axis_label = "x-axis",
                                 string y_axis_label = "y-axis",
                                 color  clr = clrDodgerBlue,
                                 bool   points_fill = true
                              )
  {
   
   ArrayResize(m_plot_names,ArraySize(m_plot_names)+1);
   m_plot_names[ArraySize(m_plot_names)-1] = plot_name;
   ChartSetInteger(m_chart_id, CHART_SHOW, m_chart_show);
   
   if(!graph.Create(m_chart_id, plot_name, m_subwin, m_x1, m_y1, m_x2, m_y2))
     {
      printf("Failed to Create graphical object on the Main chart Err = %d", GetLastError());
      return(false);
     }

//---
 
   graph.CurveAdd(x, y, clr, CURVE_POINTS_AND_LINES, legend);

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
//|                                                                  |
//+------------------------------------------------------------------+
void CPlots::PlotConfigs(long chart_id=0, int sub_win=0 ,int x1=30, int y1=70, int x2=550, int y2=350, bool chart_show=true)
 {
   m_chart_id = chart_id;
   m_subwin = sub_win;
   m_x1 = x1;
   m_y1 = y1;
   m_x2 = x2;
   m_y2 = y2;   
   m_chart_show = chart_show;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
