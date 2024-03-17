//+------------------------------------------------------------------+
//|                                                   optimizers.mqh |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"

//+------------------------------------------------------------------+
//|Class containing optimizers for updating neural network parameters|
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//| Adam (Adaptive Moment Estimation): Adam is an adaptive learning  |
//| rate optimizer that maintains learning rates for each parameter  |
//| and adapts them based on the first and second moments of the     |
//| gradients. It's known for its fast convergence and robustness to |
//| noisy gradients                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

/*enum Optimizer {
    OPTIMIZER_SGD, //Stochastic Gradient Descent
    OPTIMIZER_MiniBatchGD, //Mini-Batch Gradiend Descent
    OPTIMIZER_Adam, //Adaptive Moment Estimation
    OPTIMIZER_RMSprop, //Root Mean Square Propagation
    OPTIMIZER_Adagrad, //Adaptive Gradient Descent 
    OPTIMIZER_Adadelta, //Adadelta
    OPTIMIZER_Nadam //Nesterov-accelerated Adaptive Moment Estimation
};
*/

class OptimizerAdam
  {
protected:
   int time_step;
   double m_learning_rate;
   double m_beta1;
   double m_beta2;
   double m_epsilon;
   
   matrix moment; //first moment estimate
   matrix cache; //second moment estimate
   
public:
                     OptimizerAdam(double learning_rate=0.01, double beta1=0.9, double beta2=0.999, double epsilon=1e-8);
                    ~OptimizerAdam(void);
                    
                    virtual void update(matrix &parameters, matrix &gradients);
  };
//+------------------------------------------------------------------+
//|  Initializes the Adam optimizer with hyperparameters.            |   
//|                                                                  |
//|  learning_rate: Step size for parameter updates                  |
//|  beta1: Decay rate for the first moment estimate                 |
//|     (moving average of gradients).                               |
//|  beta2: Decay rate for the second moment estimate                |
//|     (moving average of squared gradients).                       |
//|  epsilon: Small value for numerical stability.                   |
//+------------------------------------------------------------------+
OptimizerAdam::OptimizerAdam(double learning_rate=0.010000, double beta1=0.9, double beta2=0.999, double epsilon=1e-8):
 time_step(0),
 m_learning_rate(learning_rate),
 m_beta1(beta1),
 m_beta2(beta2),
 m_epsilon(epsilon)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerAdam::~OptimizerAdam(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OptimizerAdam::update(matrix &parameters,matrix &gradients)
 {
    // Initialize moment and cache matrices if not already initialized
    if (moment.Rows() != parameters.Rows() || moment.Cols() != parameters.Cols())
    {
        moment.Resize(parameters.Rows(), parameters.Cols());
        moment.Fill(0.0);
    }

    if (cache.Rows() != parameters.Rows() || cache.Cols() != parameters.Cols())
    {
        cache.Resize(parameters.Rows(), parameters.Cols());
        cache.Fill(0.0);
    }

   
    this.time_step++; 
    
    this.moment = this.m_beta1 * this.moment + (1 -  this.m_beta1) * gradients;
    
    this.cache = this.m_beta2 * this.cache + (1 -  this.m_beta2) * MathPow(gradients, 2);

//--- Bias correction

    matrix moment_hat = this.moment / (1 - MathPow(this.m_beta1, this.time_step));
    
    matrix cache_hat = this.cache / (1 - MathPow(this.m_beta2, this.time_step));
    
    parameters -= (this.m_learning_rate * moment_hat) / (MathPow(cache_hat, 0.5) + this.m_epsilon);
 }
 
 
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//| Stochastic Gradient Descent (SGD):                               |
//|                                                                  |
//| This is the simplest optimizer                                   |
//| It updates the parameters using the gradients of the loss        |
//| function computed on individual training samples or mini-batches.|
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+


class OptimizerSGD
  {
protected:
   double m_learning_rate;
   
public:
                     OptimizerSGD(double learning_rate=0.01);
                    ~OptimizerSGD(void);
                     
                    virtual void update(matrix &parameters, matrix &gradients);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerSGD::OptimizerSGD(double learning_rate=0.01):
 m_learning_rate(learning_rate)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerSGD::~OptimizerSGD(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OptimizerSGD::update(matrix &parameters, matrix &gradients)
 {
    parameters -= this.m_learning_rate * gradients;
 }
 
 
//+------------------------------------------------------------------+
//|  Batch Gradient Descent (BGD): This optimizer computes the       |
//|  gradients of the loss function on the entire training dataset   |
//|  and updates the parameters accordingly. It can be slow and      |
//|  memory-intensive for large datasets but tends to provide a      |
//|  stable convergence.                                             |
//+------------------------------------------------------------------+


class OptimizerMinBGD: public OptimizerSGD
  {
public:
                     OptimizerMinBGD(double learning_rate=0.01);
                    ~OptimizerMinBGD(void);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerMinBGD::OptimizerMinBGD(double learning_rate=0.010000): OptimizerSGD(learning_rate)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerMinBGD::~OptimizerMinBGD(void)
 {
 
 }
 
//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|                                                                  |
//| RMSprop (Root Mean Square Propagation): RMSprop is similar to    |
//| Adam but only uses the first moment of the gradients to adapt the|
//| moment learning rates. It's effective in handling non-stationary      |
//| objectives and can converge quickly for many problems.           |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+


class OptimizerRMSprop
  {
protected:
   double m_learning_rate;
   double m_decay_rate;
   double m_epsilon;
   
   matrix<double> cache;
   
   //Dividing double/matrix causes compilation error | this is the fix to the issue
   matrix divide(const double numerator, const matrix &denominator)
    {
      matrix res = denominator;
      
      for (ulong i=0; i<denominator.Rows(); i++)
        res.Row(numerator / denominator.Row(i), i);
     return res;
    }
    
public:
                     OptimizerRMSprop(double learning_rate=0.01, double decay_rate=0.9, double epsilon=1e-8);
                    ~OptimizerRMSprop(void);
                    
                    virtual void update(matrix& parameters, matrix& gradients);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerRMSprop::OptimizerRMSprop(double learning_rate=0.01, double decay_rate=0.9, double epsilon=1e-8):
 m_learning_rate(learning_rate),
 m_decay_rate(decay_rate),
 m_epsilon(epsilon)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerRMSprop::~OptimizerRMSprop(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OptimizerRMSprop::update(matrix &parameters,matrix &gradients)
 {
 
   if (cache.Rows()!=parameters.Rows() || cache.Cols()!=parameters.Cols())
    {
     cache.Init(parameters.Rows(), parameters.Cols());
     cache.Fill(0.0);
    }
     
//---
    
    cache += m_decay_rate * cache + (1 - m_decay_rate) * MathPow(gradients, 2);
    parameters -= divide(m_learning_rate, cache + m_epsilon) * gradients;
 }


//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|                                                                  |
//| Adagrad (Adaptive Gradient Algorithm):                           |
//|                                                                  |
//| Adagrad adapts the learning rates for each parameter based on the|
//| historical gradients. It performs larger updates for infrequent  |
//| parameters and smaller updates for frequent parameters, making   |
//| it suitable for sparse data.                                     |
//|                                                                  |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

class OptimizerAdaGrad
  {
protected:
 double m_learning_rate;
 double m_epsilon;
 matrix cache;
 
 
   //Dividing double/matrix causes compilation error | this is the fix to the issue
   matrix divide(const double numerator, const matrix &denominator)
    {
      matrix res = denominator;
      
      for (ulong i=0; i<denominator.Rows(); i++)
        res.Row(numerator / denominator.Row(i), i);
     return res;
    }
 
public:
                     OptimizerAdaGrad(double learning_rate=0.01, double epsilon=1e-8);
                    ~OptimizerAdaGrad(void);
                    
                    virtual void update(matrix &parameters, matrix &gradients);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerAdaGrad::OptimizerAdaGrad(double learning_rate=0.01,double epsilon=1e-8):
 m_learning_rate(learning_rate),
 m_epsilon(epsilon)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerAdaGrad::~OptimizerAdaGrad(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OptimizerAdaGrad::update(matrix &parameters,matrix &gradients)
 {
   if (cache.Rows()!=parameters.Rows() || cache.Cols()!=parameters.Cols())
    {
     cache.Resize(parameters.Rows(), parameters.Cols());
     cache.Fill(0.0);
    }
     
//--- 
   
    cache += MathPow(gradients, 2);
    parameters -= divide(this.m_learning_rate,  MathSqrt(cache + this.m_epsilon)) * gradients;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//| Adadelta:                                                        |      
//|                                                                  |
//| Adadelta is an extension of Adagrad that aims to address         |
//| its tendency to decrease the learning rate over time. It uses a  |
//| more sophisticated update rule that accounts for the accumulated |
//| gradients over a window of time.                                 |
//|                                                                  |
//+------------------------------------------------------------------+


class OptimizerAdaDelta
  {
protected:
   double m_decay_rate;
   double m_epsilon, m_gamma, lr;
   matrix cache; //Exponential moving average of squared gradients
   
public:
                     OptimizerAdaDelta(double learning_rate=0.01, double decay_rate=0.95, double gamma=0.9, double epsilon=1e-8);
                    ~OptimizerAdaDelta(void);
                    
                     virtual void update(matrix &parameters, matrix &gradients);
  };
//+------------------------------------------------------------------+
//|   decay_rate: Decay rate for the EMA of squared deltas           |
//|   epsilons: Smoothing term to avoid division by zero             |
//|   gamma: Momentum coefficient (hyperparameter)                   |
//+------------------------------------------------------------------+
OptimizerAdaDelta::OptimizerAdaDelta(double learning_rate=0.01, double decay_rate=0.95, double gamma=0.9, double epsilon=1e-8):
  m_decay_rate(decay_rate),
  m_epsilon(epsilon),
  m_gamma(gamma),
  lr(learning_rate)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerAdaDelta::~OptimizerAdaDelta(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OptimizerAdaDelta::update(matrix &parameters, matrix &gradients)
 {
    // Initialize moment and cache matrices if not already initialized
    if (cache.Rows() != parameters.Rows() || cache.Cols() != parameters.Cols())
    {
        cache.Resize(parameters.Rows(), parameters.Cols());
        cache.Fill(0.0);
    }

//---
   
   this.cache = m_decay_rate * this.cache + (1 - m_decay_rate) * MathPow(gradients, 2);
   
   matrix delta = lr * sqrt(this.cache + m_epsilon) / sqrt(pow(gradients, 2) + m_epsilon); //Adaptive learning rate
   
   matrix momentum_term = this.m_gamma * parameters + (1 - this.m_gamma) * gradients;
   
   parameters -= delta * momentum_term;
 }

//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//| Nadam (Nesterov-accelerated Adaptive Moment Estimation):         |
//|                                                                  |
//| Nadam combines Nesterov momentum with Adam optimizer, which      |
//| results in faster convergence and better generalization.         |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+



class OptimizerNadam: protected OptimizerAdam
  {
protected:
   double m_gamma;
   
    
public:
                     OptimizerNadam(double learning_rate=0.01, double beta1=0.9, double beta2=0.999, double gamma=0.9, double epsilon=1e-8);
                    ~OptimizerNadam(void);
                    
                    virtual void update(matrix &parameters, matrix &gradients);
  };
//+------------------------------------------------------------------+
//|  Initializes the Adam optimizer with hyperparameters.            |   
//|                                                                  |
//|  learning_rate: Step size for parameter updates                  |
//|  beta1: Decay rate for the first moment estimate                 |
//|     (moving average of gradients).                               |
//|  beta2: Decay rate for the second moment estimate                |
//|     (moving average of squared gradients).                       |
//|  epsilon: Small value for numerical stability.                   |
//+------------------------------------------------------------------+
OptimizerNadam::OptimizerNadam(double learning_rate=0.010000, double beta1=0.9, double beta2=0.999, double gamma=0.9, double epsilon=1e-8)
:OptimizerAdam(learning_rate, beta1, beta2, epsilon),
 m_gamma(gamma)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
OptimizerNadam::~OptimizerNadam(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OptimizerNadam::update(matrix &parameters, matrix &gradients)
{
    // Initialize moment and cache matrices if not already initialized
    if (moment.Rows() != parameters.Rows() || moment.Cols() != parameters.Cols())
    {
        moment.Resize(parameters.Rows(), parameters.Cols());
        moment.Fill(0.0);
    }

    if (cache.Rows() != parameters.Rows() || cache.Cols() != parameters.Cols())
    {
        cache.Resize(parameters.Rows(), parameters.Cols());
        cache.Fill(0.0);
    }

    this.time_step++;

    // Update moment and cache similar to Adam
    moment = m_beta1 * moment + (1 - m_beta1) * gradients;
    cache = m_beta2 * cache + (1 - m_beta2) * MathPow(gradients, 2);

    // Bias correction
    matrix moment_hat = moment / (1 - MathPow(m_beta1, time_step));
    matrix cache_hat = cache / (1 - MathPow(m_beta2, time_step));

    // Nesterov accelerated gradient
    matrix nesterov_moment = m_gamma * moment_hat + (1 - m_gamma) * gradients;

    // Update parameters
    parameters -= m_learning_rate * nesterov_moment / sqrt(cache_hat + m_epsilon);
}

 
