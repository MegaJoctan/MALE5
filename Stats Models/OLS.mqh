//+------------------------------------------------------------------+
//|                                                          OLS.mqh |
//|                                  Copyright 2023, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
//+------------------------------------------------------------------+
//| Ordinary least squares class                                     |
//+------------------------------------------------------------------+
class OLS
  {
private:
   bool   m_const_prepended;    //check column for constant in design matrix
   matrix m_exog,               //design matrix
          m_pinv,               //pseudo-inverse of matrix
          m_cov_params,         //covariance of matrix
          m_m_error,            //error matrix
          m_norm_cov_params;    //normalized covariance matrix
   vector m_endog,              //dependent variables
          m_weights,            //weights
          m_singularvalues,     //singular values of solution
          m_params,             //coefficients of regression model(solution)
          m_tvalues,            //test statistics of model
          m_bse,                //standard errors of model
          m_const_cols,         //mark constant columns in design matrix
          m_resid;              //residuals of model
   ulong  m_obs,                //number of observations
          m_model_dof,          //degrees of freedom of model
          m_resid_dof,          //degrees of freedom of residuals
          m_kconstant,          //number of constants
          m_rank;               //rank of design matrix
   double m_aic,                //Akiake information criteria
          m_bic,                //Bayesian information criteria
          m_scale,              //scale of model
          m_llf,                //loglikelihood of model
          m_sse,                //sum of squared errors
          m_rsqe,               //r-squared of model
          m_centeredtss,        //centered sum of squares
          m_uncenteredtss;      //uncentered sum of squares
   uint              m_error;              //error flag
   // private methods
   ulong             countconstants(void);
   void              scale(void);
   void              sse(void);
   void              rsqe(void);
   void              centeredtss(void);
   void              uncenteredtss(void);
   void              aic(void);
   void              bic(void);
   void              bse(void);
   void              llf(void);
   void              tvalues(void);
   void              covariance_matrix(void);


public:
   //constructor
                     OLS(void);
   //destructor
                    ~OLS(void);
   //public methods
   bool              Fit(const matrix & x, const vector &y);
   double            predict(vector &x);
   double            predict(double x);
   //get properties of OLS model
   ulong             ModelDOF(void) { if(m_error) return 0; else return m_model_dof;}
   ulong             ResidDOF(void) { if(m_error) return 0; else return m_resid_dof;}
   double            Scale(void)  { if(m_error) return EMPTY_VALUE; else return m_scale;    }
   double            Aic(void)    { if(m_error) return EMPTY_VALUE; else return m_aic;      }
   double            Bic(void)    { if(m_error) return EMPTY_VALUE; else return m_bic;    }
   double            Sse(void)    { if(m_error) return EMPTY_VALUE; else return m_sse;    }
   double            Rsqe(void)   { if(m_error) return EMPTY_VALUE; else return m_rsqe;   }
   double            C_tss(void)  { if(m_error) return EMPTY_VALUE; else return m_centeredtss;}
   double            Loglikelihood(void) { if(m_error) return EMPTY_VALUE; return m_llf; }
   vector            Tvalues(void) { if(m_error) return m_m_error.Col(0); return m_tvalues; }
   vector            Residuals(void) { if(m_error) return m_m_error.Col(0); return m_resid; }
   vector            ModelParameters(void) { if(m_error) return m_m_error.Col(0); return m_params; }
   vector            Bse(void) { if(m_error) return m_m_error.Col(0);  return m_bse; } 
   matrix            CovarianceMatrix(void) { if(m_error) return m_m_error; return m_cov_params; }
  };
//+------------------------------------------------------------------+
//|  constructor                                                     |
//+------------------------------------------------------------------+
OLS::OLS(void)
  {
   m_kconstant = 0;
   m_obs = 0;
   m_model_dof = 0;
   m_resid_dof = 0;
   m_kconstant = 0;
   m_rank = 0;
   m_aic = 0;
   m_bic = 0;
   m_scale = 0;
   m_llf   = 0;
   m_error = 1;

   m_m_error.Resize(2,2);
   m_m_error.Fill(EMPTY_VALUE);

  }
//+------------------------------------------------------------------+
//|Destructor                                                        |
//+------------------------------------------------------------------+
OLS::~OLS(void)
  {

  }
//+------------------------------------------------------------------+
//| Fit data                                                         |
//+------------------------------------------------------------------+
bool OLS::Fit(const matrix & x, const vector &y)
  {
//---re-initialize internal properties
   m_endog = y;
//---
   m_exog  = x;
//---
   m_kconstant = countconstants();
//---
   m_error = 0;
//---
   m_model_dof = m_resid_dof = m_rank = m_obs= 0;
//---
   m_aic = m_bic= m_llf=0;
//---
   m_obs=m_exog.Rows();
//---
   m_weights.Resize(m_obs);
//---
   m_weights.Fill(1.0);
//---
   m_rank = m_exog.Rank();
//---
   m_model_dof = m_rank - m_kconstant;
//---
   m_resid_dof = m_obs - m_rank;
//---
   if(y.Size()!=x.Rows())
     {
      Print(__FUNCTION__," ",__LINE__," Error Invalid x Rows in x not equal to length of y");
      m_error = 1;
      return false;
     }
//---
   matrix U,V;
//---
   ::ResetLastError();
//--- SVD operation
   if(!m_exog.SVD(U,V,m_singularvalues))
     {
      Print(__FUNCTION__," ",__LINE__," SVD operation failed : ",::GetLastError());
      m_error = 1;
      return false;
     }
//--- Compute the pseudo-inverse of a matrix by the Moore-Penrose method
   m_pinv = m_exog.PInv();
//--- transpose m_pinv
   matrix pinv_t = m_pinv.Transpose();
//--- normalize covariance matrix
   m_norm_cov_params = m_pinv.MatMul(pinv_t);
//---
   matrix diag;
   if(!diag.Diag(m_singularvalues))
     {
      Print(__FUNCTION__," ",__LINE__," Diag operation failed : ",::GetLastError());
      m_error = 1;
      return false;
     }
//--- rank of diag
   m_rank = diag.Rank();
//--- estimate parameters of the model
   m_params = m_pinv.MatMul(m_endog);
//--- caluclate its degrees of freedom
   m_model_dof = m_rank - m_kconstant;
//--- as well as those of the residuals
   m_resid_dof = m_obs - m_rank;
//---get the residuals
   m_resid = m_endog - m_exog.MatMul(m_params);
//---compute scale of model
   scale();
//--- compute covariance matrix
   covariance_matrix();
//--- compute standard errors
   bse();
//--- compute sum of squared errors
   sse();
//--- compute loglikelihood function
   llf();
//--- compute centered sum of squares
   centeredtss();
//--- compute uncentered sum of squares
   uncenteredtss();
//--- compute r-squared
   rsqe();
//--- compute AIC
   aic();
//--- compute BIC
   bic();
//--- compute tvalues
   tvalues();
//---
   return true;
  }
//+------------------------------------------------------------------+
//| predict next value based on model parameters                     |
//+------------------------------------------------------------------+
double OLS::predict(vector &x)
  {
//---
   if(m_error)
     {
      Print("Invalid model");
      return EMPTY_VALUE;
     }
//---
   if(x.Size()!=(m_params.Size()-m_kconstant))
     {
      Print("invalid x: supplied vector does not match size of model parameters");
      return EMPTY_VALUE;
     }
//---
   double prediction = 0;
//---   
   for(ulong i = 0,k = 0;i<m_const_cols.Size(); i++)
      {
       if(!m_const_cols[i])
          prediction+=(m_params[i]*x[k++]); 
       else
          prediction+=(m_params[i]);   
      }           
//---
   return prediction;
  }
//+------------------------------------------------------------------+
//| predict next value based on model parameters                     |
//+------------------------------------------------------------------+
double OLS::predict(double x)
  {
//---
   if(m_error || (m_params.Size()-m_kconstant)>1)
     {
      if(m_error)
         Print("Invalid model");
      else
         Print("invalid x: insufficient number of predictors supplied");   
      return EMPTY_VALUE;
     }
//---
  double prediction = 0;
//---
  if(m_kconstant)
      prediction=(m_const_cols[0])?m_params[0]+(m_params[1]*x):m_params[1]+(m_params[0]*x);
  else
      prediction=m_params[0]*x;    
//---
   return prediction;
  }
//+------------------------------------------------------------------+
//|Count the number of constants in RHS matrix                       |
//+------------------------------------------------------------------+
ulong OLS::countconstants(void)
  {
   ulong count = 0;
   vector temp;
   
   m_const_cols.Resize(m_exog.Cols());
   m_const_cols.Fill(0.0);
   
   for(ulong i =0; i<m_exog.Cols(); i++)
     {
      temp = m_exog.Col(i);
      if((temp.Max()-temp.Min()) == 0.0)
         {
          count++;
          m_const_cols[i]=1.0;
         }

     }
     
   return count;
  }
//+------------------------------------------------------------------+
//|scale factor for the covariance matrix                            |
//+------------------------------------------------------------------+
void OLS::scale(void)
  {
   m_scale =  m_resid.Dot(m_resid)/double(m_resid_dof);
  }
//+------------------------------------------------------------------+
//| the variance/covariance matrix                                   |
//+------------------------------------------------------------------+
void OLS::covariance_matrix(void)
  {
//---
   m_cov_params=m_norm_cov_params*m_scale;
  }
//+------------------------------------------------------------------+
//| standard errors of the parameter estimates                       |
//+------------------------------------------------------------------+
void OLS::bse(void)
  {
   m_bse=m_cov_params.Diag();
   m_bse=MathSqrt(m_bse);
  }
//+------------------------------------------------------------------+
//| sum of squared errors                                            |
//+------------------------------------------------------------------+
void OLS::sse(void)
  {
   vector sqresid=MathPow(m_resid,2);

   m_sse = sqresid.Sum();

  }
//+------------------------------------------------------------------+
//|likelihood function for the OLS model                             |
//+------------------------------------------------------------------+
void OLS::llf(void)
  {
   double obs2=double(m_obs)/2.0;
   m_llf = -obs2*log(2.0*M_PI) - obs2*log(m_sse / double(m_obs)) - obs2;
//m_llf = -1*obs2*MathLog(2.0*M_PI*m_scale) - m_sse/(2.0*m_scale);
  }
//+------------------------------------------------------------------+
//| Akaike's information criteria                                    |
//+------------------------------------------------------------------+
void OLS::aic(void)
  {
   double params = double(m_model_dof)+double(m_kconstant);

   m_aic = -2*m_llf+2*params;

  }
//+------------------------------------------------------------------+
//|Bayes' information criteria                                       |
//+------------------------------------------------------------------+
void OLS::bic(void)
  {
   double params = double(m_model_dof)+double(m_kconstant);

   m_bic = -2*m_llf+MathLog(m_obs)*params;

  }
//+------------------------------------------------------------------+
//|t-statistic for a given parameter estimate                        |
//+------------------------------------------------------------------+
void OLS::tvalues(void)
  {
   m_tvalues = m_params/m_bse;
  }
//+------------------------------------------------------------------+
//|total  sum of squares centered about the mean                     |
//+------------------------------------------------------------------+
void OLS::centeredtss(void)
  {
   vector centered_endog = m_endog - m_endog.Mean();
//---
   m_centeredtss = centered_endog.Dot(centered_endog);
  }
//+------------------------------------------------------------------+
//| The sum of the squared values of the endogenous response variable|
//+------------------------------------------------------------------+
void OLS::uncenteredtss(void)
  {
   m_uncenteredtss = m_endog.Dot(m_endog);
  }
//+------------------------------------------------------------------+
//|R-squared of the model                                            |
//+------------------------------------------------------------------+
void OLS::rsqe(void)
  {
   m_rsqe = (m_kconstant)? 1 - m_sse/m_centeredtss: 1 - m_sse/m_uncenteredtss;
  }
//+------------------------------------------------------------------+
