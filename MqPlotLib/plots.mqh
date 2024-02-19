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
#include <MALE5\MatrixExtend.mqh>

class CPlots
  {  
protected:
   CGraphic *graph;
   
   long m_chart_id;
   int m_subwin;
   int m_x1, m_x2;
   int m_y1, m_y2;
   string m_font_family;
   bool m_chart_show;
   
   string m_plot_names[];
   ENUM_CURVE_TYPE m_curve_type;
   bool GraphCreate(string plot_name);
   
public:
         CPlots(long chart_id=0, int sub_win=0 ,int x1=30, int y1=40, int x2=550, int y2=310, string font_family="Consolas", bool chart_show=true);
        ~CPlots(void);
         
         bool Plot(string plot_name, vector& x, vector& y, string label, string x_axis_label = "x-axis", string y_axis_label = "y-axis", ENUM_CURVE_TYPE curve_type=CURVE_POINTS_AND_LINES,color  clr = clrDodgerBlue, bool   points_fill = true);
         bool AddPlot(vector& x , vector& y , string label, string x_axis_label = "x-axis", string y_axis_label = "y-axis",color  clr = clrDodgerBlue);
  };
  
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPlots::CPlots(long chart_id=0, int sub_win=0 ,int x1=30, int y1=40, int x2=550, int y2=310, string font_family="Consolas", bool chart_show=true):
   m_chart_id(chart_id),
   m_subwin(sub_win),
   m_x1(x1),
   m_y1(y1),
   m_x2(x2),
   m_y2(y2),   
   m_font_family(font_family),
   m_chart_show(chart_show)
 {
   graph = new CGraphic();
   ChartRedraw(m_chart_id);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CPlots::~CPlots(void)
 {
   for (int i=0; i<ArraySize(m_plot_names); i++)
       ObjectDelete(m_chart_id,m_plot_names[i]);
   
   delete(graph);
   ChartRedraw();
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPlots::GraphCreate(string plot_name)
 {   
   ChartRedraw(m_chart_id);
   
   ArrayResize(m_plot_names,ArraySize(m_plot_names)+1);
   m_plot_names[ArraySize(m_plot_names)-1] = plot_name;
   ChartSetInteger(m_chart_id, CHART_SHOW, m_chart_show);
   
   if(!graph.Create(m_chart_id, plot_name, m_subwin, m_x1, m_y1, m_x2, m_y2))
     {
      printf("Failed to Create graphical object on the Main chart Err = %d", GetLastError());
      return(false);
     }
     
   return (true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

bool CPlots::Plot(
                  string plot_name,
                  vector& x,
                  vector& y,
                  string label,
                  string x_axis_label = "x-axis",
                  string y_axis_label = "y-axis",
                  ENUM_CURVE_TYPE curve_type=CURVE_POINTS_AND_LINES,
                  color  clr = clrDodgerBlue,
                  bool   points_fill = true
               )
  {
   
   if (!this.GraphCreate(plot_name))
     return (false);
   
//---
   
   double x_arr[], y_arr[];
   MatrixExtend::VectorToArray(x, x_arr);
   MatrixExtend::VectorToArray(y, y_arr);
   
   m_curve_type = curve_type;
   
   graph.CurveAdd(x_arr, y_arr, ColorToARGB(clr), m_curve_type, label);

   graph.XAxis().Name(x_axis_label);
   graph.XAxis().NameSize(13);
   graph.YAxis().Name(y_axis_label);
   graph.YAxis().NameSize(13);
   graph.FontSet(m_font_family, 13);
   graph.CurvePlotAll();
   graph.Update();

   return(true);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CPlots::AddPlot(vector &x,vector &y,string label,string x_axis_label="x-axis",string y_axis_label="y-axis",color clr=16748574)
 {
   double x_arr[], y_arr[];
   MatrixExtend::VectorToArray(x, x_arr);
   MatrixExtend::VectorToArray(y, y_arr);
 
   if (!graph.CurveAdd(x_arr, y_arr, ColorToARGB(clr), m_curve_type, label))
    {
      printf("%s failed to add a plot to the existing plot Err =%d",__FUNCTION__,GetLastError());
      return false;
    }

   graph.XAxis().Name(x_axis_label);
   graph.XAxis().NameSize(13);
   graph.YAxis().Name(y_axis_label);
   graph.YAxis().NameSize(13);
   graph.FontSet(m_font_family, 13);
   graph.CurvePlotAll();
   graph.Update();

   return(true);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
