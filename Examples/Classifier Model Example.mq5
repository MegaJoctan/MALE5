//+------------------------------------------------------------------+
//|                                      Classifier Model Sample.mq5 |
//|                                     Copyright 2023, Omega Joctan |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Omega Joctan"
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"

#include <MALE5\Decision Tree\tree.mqh>
#include <MALE5\preprocessing.mqh>
#include <MALE5\MatrixExtend.mqh> //helper functions for for data manipulations
#include <MALE5\metrics.mqh> //fo measuring the performance

StandardizationScaler scaler; //standardization scaler from preprocessing.mqh
CDecisionTreeClassifier *decision_tree; //a decision tree classifier model

MqlRates rates[];
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
  
//--- Model selection
   
     decision_tree = new CDecisionTreeClassifier(2, 5); //a decision tree classifier from DecisionTree class
     
//---

     vector open, high, low, close;     
     int data_size = 1000;
     
//--- Getting the open, high, low and close values for the past 1000 bars, starting from the recent closed bar of 1
     
     open.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_OPEN, 1, data_size);
     high.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_HIGH, 1, data_size);
     low.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_LOW, 1, data_size);
     close.CopyRates(Symbol(), PERIOD_D1, COPY_RATES_CLOSE, 1, data_size);
     
     matrix X(data_size, 3); //creating the x matrix 
   
//--- Assigning the open, high, and low price values to the x matrix 

     X.Col(open, 0);
     X.Col(high, 1);
     X.Col(low, 2);
     
//--- Since we are using the x variables to predict y, we choose the close price to be the target variable 
   
     vector y(data_size); 
     for (int i=0; i<data_size; i++)
       {
         if (close[i]>open[i]) //a bullish candle appeared
           y[i] = 1; //buy signal
         else
           {
             y[i] = 0; //sell signal
           } 
       }

//--- We split the data into training and testing samples for training and evaluation
 
     matrix X_train, X_test;
     vector y_train, y_test;
     
     double train_size = 0.7; //70% of the data should be used for training the rest for testing
     int random_state = 42; //we put a random state to shuffle the data so that a machine learning model understands the patterns and not the order of the dataset, this makes the model durable
      
     MatrixExtend::TrainTestSplitMatrices(X, y, X_train, y_train, X_test, y_test, train_size, random_state); // we split the x and y data into training and testing samples         
     

//--- Normalizing the independent variables
   
     X_train = scaler.fit_transform(X_train); // we fit the scaler on the training data and transform the data alltogether
     X_test = scaler.transform(X_test); // we transform the new data this way
     

//--- Training the  model
     
     decision_tree.fit(X_train, y_train); //The training function 
     
//--- Measuring predictive accuracy 
   
     vector train_predictions = decision_tree.predict_bin(X_train);
     
     Print("Training results classification report");
     Metrics::classification_report(y_train, train_predictions);

//--- Evaluating the model on out-of-sample predictions
     
     vector test_predictions = decision_tree.predict_bin(X_test);
     
     Print("Testing results classification report");
     Metrics::classification_report(y_test, test_predictions); 
     
//---

    ArraySetAsSeries(rates, true);
        

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
    delete (decision_tree); //We have to delete the AI model object from the memory
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
     
//--- Making predictions live from the market 
   
   CopyRates(Symbol(), PERIOD_D1, 1, 3, rates); //Get the very recent information from the market
   
   vector x = {rates[0].open, rates[0].high, rates[0].low}; //Assigning data from the recent candle in a similar way to the training data
   
   x = scaler.transform(x);
   int signal = (int)decision_tree.predict_bin(x);
   
   Comment("Signal = ",signal==1?"BUY":"SELL");  //Ternary operator for checking if the signal is either buy or sell
  }
//+------------------------------------------------------------------+
