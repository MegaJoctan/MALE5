//+------------------------------------------------------------------+
//|                                                          ADF.mqh |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#include<Math\Stat\Math.mqh>
#include<Math\Alglib\specialfunctions.mqh>
#include "OLS.mqh"
#define SQRTEPS 1.4901161193847656e-08
//+------------------------------------------------------------------+
//| Information criterion                                            |
//+------------------------------------------------------------------+
enum ENUM_INFO_CRIT
  {
   INFO_NONE=0,
   INFO_AIC,
   INFO_BIC
  };
//+------------------------------------------------------------------+
//| Options for  trimming invalid observations                       |
//+------------------------------------------------------------------+
enum ENUM_TRIM
  {
   TRIM_NONE=0,
   TRIM_FORWARD,
   TRIM_BACKWARD,
   TRIM_BOTH
  };
//+------------------------------------------------------------------+
//| options for how to handle original data set                      |
//+------------------------------------------------------------------+

enum ENUM_ORIGINAL
  {
   ORIGINAL_EX=0,
   ORIGINAL_IN,
   ORIGINAL_SEP
  };
//+------------------------------------------------------------------+
//| Constant and trend used in regression model                      |
//+------------------------------------------------------------------+

enum ENUM_TREND
  {
   TREND_NONE=0,
   TREND_CONST_ONLY,
   TREND_LINEAR_ONLY,
   TREND_LINEAR_CONST,
   TREND_QUAD_LINEAR_CONST
  };
//+------------------------------------------------------------------+
//| Options for how to handle existing constants                     |
//+------------------------------------------------------------------+

enum ENUM_HAS_CONST
  {
   HAS_CONST_RAISE=0,
   HAS_CONST_SKIP,
   HAS_CONST_ADD
  };
//+------------------------------------------------------------------+
//|helper function : adds selected trend and\or constant             |
//+------------------------------------------------------------------+
bool addtrend(matrix &in,matrix &out,ENUM_TREND trend=TREND_CONST_ONLY, bool prepend=false, ENUM_HAS_CONST has_const=HAS_CONST_SKIP)
  {
//---
   ulong trendorder=0;
//---
   if(trend==TREND_NONE)
      return out.Copy(in);
//---
   if(trend==TREND_CONST_ONLY)
      trendorder = 0;
   else
      if(trend==TREND_LINEAR_CONST || trend==TREND_LINEAR_ONLY)
         trendorder = 1;
      else
         if(trend==TREND_QUAD_LINEAR_CONST)
            trendorder = 2;
//---
   ulong nobs = in.Rows();
//---
   vector tvector,temp;
//---
   if(!tvector.Resize(nobs) || !temp.Resize(nobs))
     {
      Print(__FUNCTION__," ",__LINE__," Resize Error ", ::GetLastError());
      return false;
     }
//---
   double sequence[];
//---
   if(!MathSequence(1,nobs,1,sequence) || !tvector.Assign(sequence))
     {
      Print(__FUNCTION__," ",__LINE__," Error ", ::GetLastError());
      return false;
     }
//---
   matrix trendmat;
//---
   if(!trendmat.Resize(tvector.Size(),(trend==TREND_LINEAR_ONLY)?1:trendorder+1))
     {
      Print(__FUNCTION__," ",__LINE__," Resize Error ", ::GetLastError());
      return false;
     }
//---
   for(ulong i = 0; i<trendmat.Cols(); i++)
     {
      temp = MathPow(tvector,(trend==TREND_LINEAR_ONLY)?double(i+1):double(i));
      if(!trendmat.Col(temp,i))
        {
         Print(__FUNCTION__," ",__LINE__," Matrix Assign Error ", ::GetLastError());
         return false;
        }
     }
//---
   vector ptp = in.Ptp(0);
//---
   if(!ptp.Min())
     {
      if(has_const==HAS_CONST_RAISE)
        {
         Print("Input matrix contains one or more constant columns");
         return false;
        }
      if(has_const==HAS_CONST_SKIP)
        {
         matrix vsplitted[];

         ulong parts[] = {1,trendmat.Cols()-1};

         trendmat.Vsplit(parts,vsplitted);

         if(!trendmat.Copy(vsplitted[1]))
           {
            Print(__FUNCTION__," ",__LINE__," Resize Error ", ::GetLastError());
            return false;
           }
        }
     }
//---
   int order = (prepend)?1:-1;
//---
   if(!out.Resize(trendmat.Rows(),trendmat.Cols()+in.Cols()))
     {
      Print(__FUNCTION__," ",__LINE__," Resize Error ", ::GetLastError());
      return false;
     }
//---
   ulong j = 0;
   matrix mtemp;
//---
   if(prepend)
      mtemp.Copy(trendmat);
   else
      mtemp.Copy(in);
//---
   for(j = 0; j<mtemp.Cols(); j++)
     {
      vector col = mtemp.Col(j);

      if(!out.Col(col,j))
        {
         Print(__FUNCTION__," ",__LINE__," Matrix Assign Error ", ::GetLastError());
         return false;
        }

     }
//---
   ulong minus = j;
//---
   if(prepend)
      mtemp.Copy(in);
   else
      mtemp.Copy(trendmat);
//---
   for(; j<out.Cols(); j++)
     {
      vector col = mtemp.Col(j-minus);

      if(!out.Col(col,j))
        {
         Print(__FUNCTION__," ",__LINE__," Matrix Assign Error ", ::GetLastError());
         return false;
        }

     }
//---
   return true;
//---
  }

//+----------------------------------------------------------------------+
//| calculates MacKinnon's approximate p-value for a given test statistic|
//+----------------------------------------------------------------------+
double mackinnonp(double teststat, ENUM_TREND trend = TREND_CONST_ONLY,ulong nseries = 1, uint lags =0)
  {
   vector small_scaling =  {1, 1, 1e-2};
   vector large_scaling =  {1, 1e-1, 1e-1, 1e-2};

   double tau_star_nc []= {-1.04, -1.53, -2.68, -3.09, -3.07, -3.77};
   double tau_min_nc []= {-19.04, -19.62, -21.21, -23.25, -21.63, -25.74};
   double tau_max_nc []= {double("inf"), 1.51, 0.86, 0.88, 1.05, 1.24};
   double tau_star_c []= {-1.61, -2.62, -3.13, -3.47, -3.78, -3.93};
   double tau_min_c []= {-18.83, -18.86, -23.48, -28.07, -25.96, -23.27};
   double tau_max_c []= {2.74, 0.92, 0.55, 0.61, 0.79, 1};
   double tau_star_ct []= {-2.89, -3.19, -3.50, -3.65, -3.80, -4.36};
   double tau_min_ct []= {-16.18, -21.15, -25.37, -26.63, -26.53, -26.18};
   double tau_max_ct []= {0.7, 0.63, 0.71, 0.93, 1.19, 1.42};
   double tau_star_ctt []= {-3.21, -3.51, -3.81, -3.83, -4.12, -4.63};
   double tau_min_ctt []= {-17.17, -21.1, -24.33, -24.03, -24.33, -28.22};
   double tau_max_ctt []= {0.54, 0.79, 1.08, 1.43, 3.49, 1.92};

   double tau_nc_smallp [][3]=
     {
        {0.6344, 1.2378, 3.2496},
        {1.9129, 1.3857, 3.5322},
        {2.7648, 1.4502, 3.4186},
        {3.4336, 1.4835, 3.19},
        {4.0999, 1.5533, 3.59},
        {4.5388, 1.5344, 2.9807}
     };

   double tau_c_smallp [][3]=
     {
        {2.1659, 1.4412, 3.8269},
        {2.92, 1.5012, 3.9796},
        {3.4699, 1.4856, 3.164},
        {3.9673, 1.4777, 2.6315},
        {4.5509, 1.5338, 2.9545},
        {5.1399, 1.6036, 3.4445}
     };

   double tau_ct_smallp [][3]=
     {
        {3.2512, 1.6047, 4.9588},
        {3.6646, 1.5419, 3.6448},
        {4.0983, 1.5173, 2.9898},
        {4.5844, 1.5338, 2.8796},
        {5.0722, 1.5634, 2.9472},
        {5.53, 1.5914, 3.0392}
     };

   double tau_ctt_smallp [][3]=
     {
        {4.0003, 1.658, 4.8288},
        {4.3534, 1.6016, 3.7947},
        {4.7343, 1.5768, 3.2396},
        {5.214, 1.6077, 3.3449},
        {5.6481, 1.6274, 3.3455},
        {5.9296, 1.5929, 2.8223}
     };



   double tau_nc_largep [][4]=
     {
        {0.4797, 9.3557, -0.6999, 3.3066},
        {1.5578, 8.558, -2.083, -3.3549},
        {2.2268, 6.8093, -3.2362, -5.4448},
        {2.7654, 6.4502, -3.0811, -4.4946},
        {3.2684, 6.8051, -2.6778, -3.4972},
        {3.7268, 7.167, -2.3648, -2.8288}
     };

   double tau_c_largep [][4]=
     {
        {1.7339, 9.3202, -1.2745, -1.0368},
        {2.1945, 6.4695, -2.9198, -4.2377},
        {2.5893, 4.5168, -3.6529, -5.0074},
        {3.0387, 4.5452, -3.3666, -4.1921},
        {3.5049, 5.2098, -2.9158, -3.3468},
        {3.9489, 5.8933, -2.5359, -2.721}
     };

   double tau_ct_largep [][4]=
     {
        {2.5261, 6.1654, -3.7956, -6.0285},
        {2.85, 5.272, -3.6622, -5.1695},
        {3.221, 5.255, -3.2685, -4.1501},
        {3.652, 5.9758, -2.7483, -3.2081},
        {4.0712, 6.6428, -2.3464, -2.546},
        {4.4735, 7.1757, -2.0681, -2.1196}
     };

   double tau_ctt_largep [][4]=
     {
        {3.0778, 4.9529, -4.1477, -5.9359},
        {3.4713, 5.967, -3.2507, -4.2286},
        {3.8637, 6.7852, -2.6286, -3.1381},
        {4.2736, 7.6199, -2.1534, -2.4026},
        {4.6679, 8.2618, -1.822, -1.9147},
        {5.0009, 8.3735, -1.6994, -1.6928}
     };


   vector maxstat,minstat,starstat;
   matrix tau_smallps, tau_largeps;

   switch(trend)
     {
      case TREND_NONE:
         if(!maxstat.Assign(tau_max_nc) ||
            !minstat.Assign(tau_min_nc) ||
            !starstat.Assign(tau_star_nc)||
            !tau_smallps.Assign(tau_nc_smallp)||
            !tau_largeps.Assign(tau_nc_largep))
           {
            Print("assignment error :", GetLastError());
            return double("inf");
           }
         else
            break;
      case TREND_CONST_ONLY:
         if(!maxstat.Assign(tau_max_c) ||
            !minstat.Assign(tau_min_c) ||
            !starstat.Assign(tau_star_c)||
            !tau_smallps.Assign(tau_c_smallp)||
            !tau_largeps.Assign(tau_c_largep))
           {
            Print("assignment error :", GetLastError());
            return double("inf");
           }
         else
            break;
      case TREND_LINEAR_CONST:
         if(!maxstat.Assign(tau_max_ct) ||
            !minstat.Assign(tau_min_ct) ||
            !starstat.Assign(tau_star_ct)||
            !tau_smallps.Assign(tau_ct_smallp)||
            !tau_largeps.Assign(tau_ct_largep))
           {
            Print("assignment error :", GetLastError());
            return double("inf");
           }
         else
            break;
      case TREND_QUAD_LINEAR_CONST:
         if(!maxstat.Assign(tau_max_ctt) ||
            !minstat.Assign(tau_min_ctt) ||
            !starstat.Assign(tau_star_ctt)||
            !tau_smallps.Assign(tau_ctt_smallp)||
            !tau_largeps.Assign(tau_ctt_largep))
           {
            Print("assignment error :", GetLastError());
            return double("inf");
           }
         else
            break;
      default:
         Print(__FUNCTION__," Error invalid input for trend argument");
         return double("nan");
     }

   if(teststat>maxstat[nseries-1])
      return 1.0;
   else
      if(teststat<minstat[nseries-1])
         return 0.0;


   vector tau_coef;

   if(teststat<=starstat[nseries-1])
      tau_coef = small_scaling*(tau_smallps.Row(nseries-1));
   else
      tau_coef = large_scaling*(tau_largeps.Row(nseries-1));


   double rv,tau[];

   ArrayResize(tau,int(tau_coef.Size()));

   for(ulong i=0; i<tau_coef.Size(); i++)
      tau[i]=tau_coef[tau_coef.Size()-1-i];

   rv=polyval(tau,teststat);

   return CNormalDistr::NormalCDF(rv);
  }

//+------------------------------------------------------------------+
//| helper function : evaluates a polynomial                         |
//+------------------------------------------------------------------+
double polyval(double &in_array[], double coeff)
  {
   double retv = 0;

   for(uint i = 0; i<in_array.Size(); i++)
      retv = retv * coeff + in_array[i];

   return retv;
  }
//+------------------------------------------------------------------+
//|Computes critical values                                          |
//+------------------------------------------------------------------+
vector mackinnoncrit(ulong nseries = 1,ENUM_TREND trend = TREND_CONST_ONLY, ulong num_obs=ULONG_MAX)
  {
   matrix tau_nc_2010 [] = {{
           {-2.56574, -2.2358, -3.627, 0},  // N [] = 1
           {-1.94100, -0.2686, -3.365, 31.223},
           {-1.61682, 0.2656, -2.714, 25.364}
        }
     };

   matrix tau_c_2010 [] =
     {
        {  {-3.43035, -6.5393, -16.786, -79.433},  // N [] = 1, 1%
           {-2.86154, -2.8903, -4.234, -40.040},   // 5 %
           {-2.56677, -1.5384, -2.809, 0}
        },        // 10 %
        {  {-3.89644, -10.9519, -33.527, 0},       // N [] = 2
           {-3.33613, -6.1101, -6.823, 0},
           {-3.04445, -4.2412, -2.720, 0}
        },
        {  {-4.29374, -14.4354, -33.195, 47.433},  // N [] = 3
           {-3.74066, -8.5632, -10.852, 27.982},
           {-3.45218, -6.2143, -3.718, 0}
        },
        {  {-4.64332, -18.1031, -37.972, 0},       // N [] = 4
           {-4.09600, -11.2349, -11.175, 0},
           {-3.81020, -8.3931, -4.137, 0}
        },
        {  {-4.95756, -21.8883, -45.142, 0},       // N [] = 5
           {-4.41519, -14.0405, -12.575, 0},
           {-4.13157, -10.7417, -3.784, 0}
        },
        {  {-5.24568, -25.6688, -57.737, 88.639},  // N [] = 6
           {-4.70693, -16.9178, -17.492, 60.007},
           {-4.42501, -13.1875, -5.104, 27.877}
        },
        {  {-5.51233, -29.5760, -69.398, 164.295},  // N [] = 7
           {-4.97684, -19.9021, -22.045, 110.761},
           {-4.69648, -15.7315, -5.104, 27.877}
        },
        {  {-5.76202, -33.5258, -82.189, 256.289},  // N [] = 8
           {-5.22924, -23.0023, -24.646, 144.479},
           {-4.95007, -18.3959, -7.344, 94.872}
        },
        {  {-5.99742, -37.6572, -87.365, 248.316},  // N [] = 9
           {-5.46697, -26.2057, -26.627, 176.382},
           {-5.18897, -21.1377, -9.484, 172.704}
        },
        {  {-6.22103, -41.7154, -102.680, 389.33},  // N [] = 10
           {-5.69244, -29.4521, -30.994, 251.016},
           {-5.41533, -24.0006, -7.514, 163.049}
        },
        {  {-6.43377, -46.0084, -106.809, 352.752},  // N [] = 11
           {-5.90714, -32.8336, -30.275, 249.994},
           {-5.63086, -26.9693, -4.083, 151.427}
        },
        {  {-6.63790, -50.2095, -124.156, 579.622},  // N [] = 12
           {-6.11279, -36.2681, -32.505, 314.802},
           {-5.83724, -29.9864, -2.686, 184.116}
        }
     };

   matrix tau_ct_2010 [] =
     {
        {  {-3.95877, -9.0531, -28.428, -134.155},   // N [] = 1
           {-3.41049, -4.3904, -9.036, -45.374},
           {-3.12705, -2.5856, -3.925, -22.380}
        },
        {  {-4.32762, -15.4387, -35.679, 0},         // N [] = 2
           {-3.78057, -9.5106, -12.074, 0},
           {-3.49631, -7.0815, -7.538, 21.892}
        },
        {  {-4.66305, -18.7688, -49.793, 104.244},   // N [] = 3
           {-4.11890, -11.8922, -19.031, 77.332},
           {-3.83511, -9.0723, -8.504, 35.403}
        },
        {  {-4.96940, -22.4694, -52.599, 51.314},    // N [] = 4
           {-4.42871, -14.5876, -18.228, 39.647},
           {-4.14633, -11.2500, -9.873, 54.109}
        },
        {  {-5.25276, -26.2183, -59.631, 50.646},    // N [] = 5
           {-4.71537, -17.3569, -22.660, 91.359},
           {-4.43422, -13.6078, -10.238, 76.781}
        },
        {  {-5.51727, -29.9760, -75.222, 202.253},   // N [] = 6
           {-4.98228, -20.3050, -25.224, 132.03},
           {-4.70233, -16.1253, -9.836, 94.272}
        },
        {  {-5.76537, -33.9165, -84.312, 245.394},   // N [] = 7
           {-5.23299, -23.3328, -28.955, 182.342},
           {-4.95405, -18.7352, -10.168, 120.575}
        },
        {  {-6.00003, -37.8892, -96.428, 335.92},    // N [] = 8
           {-5.46971, -26.4771, -31.034, 220.165},
           {-5.19183, -21.4328, -10.726, 157.955}
        },
        {  {-6.22288, -41.9496, -109.881, 466.068},  // N [] = 9
           {-5.69447, -29.7152, -33.784, 273.002},
           {-5.41738, -24.2882, -8.584, 169.891}
        },
        {  {-6.43551, -46.1151, -120.814, 566.823},  // N [] = 10
           {-5.90887, -33.0251, -37.208, 346.189},
           {-5.63255, -27.2042, -6.792, 177.666}
        },
        {  {-6.63894, -50.4287, -128.997, 642.781},  // N [] = 11
           {-6.11404, -36.4610, -36.246, 348.554},
           {-5.83850, -30.1995, -5.163, 210.338}
        },
        {  {-6.83488, -54.7119, -139.800, 736.376},  // N [] = 12
           {-6.31127, -39.9676, -37.021, 406.051},
           {-6.03650, -33.2381, -6.606, 317.776}
        }
     };

   matrix tau_ctt_2010 [] =
     {
        {  {-4.37113, -11.5882, -35.819, -334.047},  // N [] = 1
           {-3.83239, -5.9057, -12.490, -118.284},
           {-3.55326, -3.6596, -5.293, -63.559}
        },
        {  {-4.69276, -20.2284, -64.919, 88.884},    // N [] =2
           {-4.15387, -13.3114, -28.402, 72.741},
           {-3.87346, -10.4637, -17.408, 66.313}
        },
        {  {-4.99071, -23.5873, -76.924, 184.782},   // N [] = 3
           {-4.45311, -15.7732, -32.316, 122.705},
           {-4.17280, -12.4909, -17.912, 83.285}
        },
        {  {-5.26780, -27.2836, -78.971, 137.871},   // N [] = 4
           {-4.73244, -18.4833, -31.875, 111.817},
           {-4.45268, -14.7199, -17.969, 101.92}
        },
        {  {-5.52826, -30.9051, -92.490, 248.096},   // N [] = 5
           {-4.99491, -21.2360, -37.685, 194.208},
           {-4.71587, -17.0820, -18.631, 136.672}
        },
        {  {-5.77379, -34.7010, -105.937, 393.991},  // N [] = 6
           {-5.24217, -24.2177, -39.153, 232.528},
           {-4.96397, -19.6064, -18.858, 174.919}
        },
        {  {-6.00609, -38.7383, -108.605, 365.208},  // N [] = 7
           {-5.47664, -27.3005, -39.498, 246.918},
           {-5.19921, -22.2617, -17.910, 208.494}
        },
        {  {-6.22758, -42.7154, -119.622, 421.395},  // N [] = 8
           {-5.69983, -30.4365, -44.300, 345.48},
           {-5.42320, -24.9686, -19.688, 274.462}
        },
        {  {-6.43933, -46.7581, -136.691, 651.38},   // N [] = 9
           {-5.91298, -33.7584, -42.686, 346.629},
           {-5.63704, -27.8965, -13.880, 236.975}
        },
        {  {-6.64235, -50.9783, -145.462, 752.228},  // N [] = 10
           {-6.11753, -37.056, -48.719, 473.905},
           {-5.84215, -30.8119, -14.938, 316.006}
        },
        {  {-6.83743, -55.2861, -152.651, 792.577},  // N [] = 11
           {-6.31396, -40.5507, -46.771, 487.185},
           {-6.03921, -33.8950, -9.122, 285.164}
        },
        {  {-7.02582, -59.6037, -166.368, 989.879},  // N [] = 12
           {-6.50353, -44.0797, -47.242, 543.889},
           {-6.22941, -36.9673, -10.868, 418.414}
        }
     };

   vector ret_vector = {0,0,0};

   switch(trend)
     {
      case TREND_CONST_ONLY:
         process(tau_c_2010,ret_vector,num_obs,nseries);
         break;
      case TREND_NONE:
         process(tau_nc_2010,ret_vector,num_obs,nseries);
         break;
      case TREND_LINEAR_CONST:
         process(tau_ct_2010,ret_vector,num_obs,nseries);
         break;
      case TREND_QUAD_LINEAR_CONST:
         process(tau_ctt_2010,ret_vector,num_obs,nseries);
         break;
      default:
         Print("Invalid input for trend argument");
         return ret_vector;
     }

   return ret_vector;
  }
//+------------------------------------------------------------------+
//|helper function: evaluates a multiple variable polynomial         |
//+------------------------------------------------------------------+
void process(matrix &tau[], vector &out, ulong numobs, ulong ns)
  {
   if(numobs==ULONG_MAX)
      out = tau[ns-1].Col(0);
   else
     {
      for(ulong i=0; i<tau[ns-1].Cols(); i++)
         out = out * (1.0/double(numobs)) + tau[ns-1].Col(tau[ns-1].Cols()-1-i);
     }
   return;
  }

  

//+---------------------------------------------------------------------+
//|Class CAdf                                                           |
//|   encapsulates the the Augmented Dickey Fuller Test for Stationarity|
//+---------------------------------------------------------------------+

class CAdf
{
 private:
  double m_adf_stat,  //adf statistic
         m_bestic,    //optimal bic or aic
         m_pvalue;    //p-value
  ulong  m_usedlag;   //lag used for optimal reg model
  vector m_critvals;  //estimated critical values
  OLS    *m_ols;      //internal ordinary least squares reg model
   // private methods
 bool   gridsearch(vector &LHS, matrix &RHS, ulong f_lag, ulong l_lag,ENUM_INFO_CRIT crit, double &b_ic, ulong &best_lag);
 bool   lagmat(matrix &in,matrix &out[],ulong mlag,ENUM_TRIM trim=TRIM_BOTH,ENUM_ORIGINAL original=ORIGINAL_IN);
 bool   prepare_lhs_rhs(vector &lhs, matrix &rhs, double &in[], double &in_diff[],ulong lag);
 
 
 public:
  CAdf(void);
  ~CAdf(void);
  
 bool Adfuller(double &array[],ulong max_lag = 0,ENUM_TREND trend = TREND_CONST_ONLY, ENUM_INFO_CRIT autolag=INFO_AIC);
 vector CriticalValues(void) {  return m_critvals; }
 double AdfStatistic(void)   {  return m_adf_stat; }
 double Pvalue(void)         {  return m_pvalue;   }
};
//+------------------------------------------------------------------+
//| Constructor                                                      |
//+------------------------------------------------------------------+
CAdf::CAdf(void)
{
 m_ols = new OLS();
 
 m_adf_stat = m_bestic = m_pvalue = EMPTY_VALUE;
 
 m_usedlag = 0;
 
 m_critvals = vector::Zeros(3);
}
//+------------------------------------------------------------------+
//| Destructor                                                       |
//+------------------------------------------------------------------+
CAdf::~CAdf(void)
{
 if(CheckPointer(m_ols)==POINTER_DYNAMIC)
    delete m_ols;

}
//+------------------------------------------------------------------+
//| Augmented Dickey-Fuller unit root test                           |
//+------------------------------------------------------------------+
bool CAdf::Adfuller(double&array[],ulong max_lag = 0,ENUM_TREND trend = TREND_CONST_ONLY, ENUM_INFO_CRIT autolag=INFO_AIC)
  {
//---  
   if(CheckPointer(m_ols)==POINTER_INVALID)
    {
     Print("Critical internal error: Invalid Pointer to OLS object");
     return false;
    }
//---
   if(array[ArrayMaximum(array)]==array[ArrayMinimum(array)])
     {
      Print(__FUNCTION__," ",__LINE__," Invalid input series is made of constants ");
      return false;
     }
//---
   uint nobs = array.Size();
//---
   uint ntrend = (trend>1)?uint(trend)-1:uint(trend);
//---
   ulong maxlag;
//---
   if(!max_lag)
     {
      maxlag =  uint(ceil(12.0 * pow(nobs / 100.0, 1.0 / 4.0)));
      maxlag =  MathMin(nobs / 2 - ntrend - 1, maxlag);

      if(ntrend>=2)
        {
         Print("sample size is too short to use selected regression component: Adjust trend input value ");
         return false;
        }
     }
   else
     {
      maxlag = max_lag;

      if(ntrend>=2)
        {
         Print("sample size is too short to use selected regression component: Adjust trend input value ");
         return false;
        }

      if(maxlag>floor(nobs/2-ntrend-1))
        {
         Print("maxlag must be less than (nobs/2 - 1 - ntrend) \n where ntrend is the number of included deterministic regressors");
         return false;
        }
     }
//---
   double diff[];
//---
   if(!MathDifference(array,1,diff))
     {
      Print(__FUNCTION__," ",__LINE__," Error differencing error : ",::GetLastError());
      return false;
     }
//---
   vector xdshort, tmp;
//---
   matrix xdiff;
//---
   if(!xdiff.Resize(diff.Size(),1) || !tmp.Resize(diff.Size()) || !tmp.Assign(diff) || !xdiff.Col(tmp,0))
     {
      Print(__FUNCTION__," ",__LINE__," Resize or assign error : ",::GetLastError());
      return false;
     }
//---
   matrix xdall[];
//---
   if(!lagmat(xdiff,xdall,maxlag))
      return false;
//---
   if(!prepare_lhs_rhs(xdshort,xdall[0],array,diff,maxlag))
      return false;

//---
   matrix fullRHS;
   double bestic;
   ulong startlag,bestlag,usedlag;
//---
   if(autolag!=INFO_NONE)
     {
      if(trend!=TREND_NONE)
        {
         if(!addtrend(xdall[0],fullRHS,trend,true))
            return false;
        }
      else
        {
         if(!fullRHS.Copy(xdall[0]))
           {
            Print(__FUNCTION__," ",__LINE__," Resize or assign error : ",::GetLastError());
            return false;
           }
        }

      startlag = fullRHS.Cols() - xdall[0].Cols() + 1;

      gridsearch(xdshort,fullRHS,startlag,maxlag,autolag,bestic,bestlag);

      bestlag-=startlag;

      if(!lagmat(xdiff,xdall,bestlag))
         return false;

      if(!prepare_lhs_rhs(xdshort,xdall[0],array,diff,bestlag))
         return false;

      usedlag = bestlag;

     }
   else
     {
      usedlag = maxlag;
      bestic = 0;
     }
//---
   m_usedlag = usedlag;
   m_bestic = bestic;
//---   
   matrix xv;
//---
   xv.Resize(xdall[0].Rows(),ulong(usedlag+1),2);
//---
   for(ulong k=0; k<ulong(usedlag+1); k++)
     {
      if(!xv.Col(xdall[0].Col(k),k))
        {
         Print(__FUNCTION__," ",__LINE__," Error adding column: ",::GetLastError());
         return false;
        }
     }
//---
   if(trend!=TREND_NONE)
     {
      if(!addtrend(xv,fullRHS,trend))
        {
         Print(__FUNCTION__," ",__LINE__," Error processing input ");
         return false;
        }
     }
   else
     {
      if(!fullRHS.Copy(xv))
        {
         Print(__FUNCTION__," ",__LINE__," Error processing input ");
         return false;
        }
     }


   if(!m_ols.Fit(fullRHS, xdshort))
     {
      Print(__FUNCTION__," ",__LINE__," Error OLS fit: ",::GetLastError());
      return false;
     }
//---
   vector adfstat = m_ols.Tvalues();
//---
   m_adf_stat = adfstat[0];
//---
   m_pvalue = mackinnonp(m_adf_stat,trend);
//---
   m_critvals = mackinnoncrit(1,trend,fullRHS.Rows());
//---
   return true;
  }
//+------------------------------------------------------------------+
//| helper function prepares rhs of equation                         |
//+------------------------------------------------------------------+
bool CAdf::prepare_lhs_rhs(vector &lhs, matrix &rhs, double &in[], double &in_diff[],ulong lag)
  {
   ulong len= rhs.Rows();

   if(!lhs.Resize(len))
     {
      Print(__FUNCTION__," ",__LINE__," Resize or assign error : ",::GetLastError());
      return false;
     }

   vector cpy;
   if(!cpy.Resize(len))
     {
      Print(__FUNCTION__," ",__LINE__," Resize or assign error : ",::GetLastError());
      return false;
     }

   ulong mm = ulong(in.Size())-(len+1);

   for(ulong i=mm; i<ulong(in.Size()-1); i++)
      cpy[i-mm] = in[i];

   if(!rhs.Col(cpy,0))
     {
      Print(__FUNCTION__," ",__LINE__," Resize or assign error : ",::GetLastError());
      return false;
     }

   mm = ulong(in_diff.Size())-(len);

   for(ulong i=mm; i<ulong(in_diff.Size()); i++)
      lhs[i-mm] = in_diff[i];

   return true;
  }
//+-----------------------------------------------------------------------------+
//|  helper function performs search for optimal lag based in selected criterion|
//+-----------------------------------------------------------------------------+
bool CAdf::gridsearch(vector &LHS, matrix &RHS, ulong f_lag, ulong l_lag,ENUM_INFO_CRIT crit, double &b_ic, ulong &best_lag)
  {
//---
   matrix xv;
//---
   b_ic=DBL_MAX;
//---
   double ic=0;
//---
   for(ulong i = f_lag; i<f_lag+l_lag+1; i++)
     {
      //---
      xv.Resize(RHS.Rows(),ulong(i),2);
      //---
      //vector v;
      //---
      for(ulong k=0; k<ulong(i); k++)
        {
         if(!xv.Col(RHS.Col(k),k))
           {
            Print(__FUNCTION__," ",__LINE__," Error adding column: ",::GetLastError());
            return false;
           }
        }

      //---
      if(!m_ols.Fit(xv, LHS))
        {
         Print(__FUNCTION__," ",__LINE__," Error OLS fit: ",::GetLastError());
         return false;
        }
      //---
      switch(crit)
        {
         case INFO_AIC:
            ic = m_ols.Aic();
            break;
         case INFO_BIC:
            ic = m_ols.Bic();
            break;
        }
      //---
      if(ic<b_ic)
        {
         b_ic = ic;
         best_lag = i;
        }
      //---
     }
//---

   return true;
  }
//+------------------------------------------------------------------+
//| helper function: transforms rhs matrix                           |
//+------------------------------------------------------------------+
bool CAdf::lagmat(matrix &in,matrix &out[],ulong mlag,ENUM_TRIM trim=TRIM_BOTH,ENUM_ORIGINAL original=ORIGINAL_IN)
  {
//---
   ulong nobs=in.Rows();
   ulong nvars =in.Cols();
   ulong dropidx = 0;
//---
   if(mlag>=nobs)
     {
      Print("mlag should be < number of rows of the input matrix");
      return false;
     }
//---
   if(original==ORIGINAL_EX || original==ORIGINAL_SEP)
      dropidx = nvars;
//---
   matrix nn;
//---
   if(!nn.Resize(nobs + mlag,nvars * (mlag + 1)))
     {
      Print(__FUNCTION__," ",__LINE__," Resize Error ", ::GetLastError());
      return false;
     }
//---
   nn.Fill(0.0);
//---
   ulong maxlag=mlag;
//---
   ulong row_end,row_start,col_end,col_start,j,z;
   row_end=row_start=col_end=col_start=j=z=0;
//---
   for(ulong k = 0; k<(maxlag+1); k++)
     {
      row_start = maxlag-k;
      row_end = nobs+maxlag-k;
      col_start = nvars*(maxlag-k);
      col_end = nvars*(maxlag-k+1);
      j = 0;
      for(ulong irow = row_start; irow < row_end; irow++, j++)
        {
         z = 0;
         for(ulong icol = col_start; icol < col_end; icol++, z++)
            nn[irow][icol]=in[j][z];
        }
     }
//---
   ulong startobs,stopobs;
   if(trim==TRIM_NONE || trim==TRIM_FORWARD)
      startobs=0;
   else
      startobs=maxlag;
//---
   if(trim==TRIM_NONE || trim==TRIM_BACKWARD)
      stopobs=nn.Rows();
   else
      stopobs=nobs;
//---
   if(dropidx)
      ArrayResize(out,2);
   else
      ArrayResize(out,1);
//---
   if(!out[0].Resize(stopobs-startobs,nn.Cols()-dropidx))
     {
      Print(__FUNCTION__," ",__LINE__," Resize Error ", ::GetLastError());
      return false;
     }
//---
   for(ulong irow = startobs; irow<stopobs; irow++)
      for(ulong icol=dropidx; icol<nn.Cols(); icol++)
         out[0][irow-startobs][icol-dropidx]=nn[irow][icol];
//---
   if(out.Size()>1)
     {
      if(!out[1].Resize(stopobs-startobs,dropidx))
        {
         Print(__FUNCTION__," ",__LINE__," Resize Error ", ::GetLastError());
         return false;
        }
      //---
      for(ulong irow = startobs; irow<stopobs; irow++)
         for(ulong icol=0; icol<out[1].Cols(); icol++)
            out[1][irow-startobs][icol]=nn[irow][icol];
     }
//---
   return true;
  }
