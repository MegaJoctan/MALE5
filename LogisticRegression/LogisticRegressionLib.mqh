//hire me on your next big Machine Learning Project on this link > https://www.mql5.com/en/job/new?prefered=omegajoctan
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

//don't forget to change this directory

#include "C:\Users\Omega Joctan\AppData\Roaming\MetaQuotes\Terminal\892B47EBC091D6EF95E3961284A76097\MQL5\Experts\DataScience\LinearRegression\LinearRegressionLib.mqh";

//---
class CLogisticRegression: public CMatrixRegression

  {
     private: 
                           double  e;  //Euler's number 
                           
                           string  m_AllDataArrayString[];
                           double  m_AllDataArrayDouble[];
                           
                           string  ColumnsToEncode[];
                           string  ColumnsToFix[];
                           string  MembersToEncode[];
                           
                           double  m_yintercept;    //constant for all values of x 
                           int     each_arrsize;
                           
     protected:             
                           int     single_rowtotal; 
                           
                           void    FixMissingValuesFunction(double& Arr[],int column_number,int index);
                           void    EncodeLabelFunction(string& Arr[],double& output_arr[], int column_number, int index);
                           void    WriteToCSV(int &Predicted[],string &date_time[],string file_name,string delimiter=",");
                           
     public:
                           CLogisticRegression(void);
                          ~CLogisticRegression(void);
                           
                           //These should be called before the Init 
                           
                           void    FixMissingValues(string columns);
                           void    LabelEncoder(string columns, string members);
                           
                           //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                           
                           void    Init(string filename=NULL,string delimiter=",",int y_column=1,string x_columns="3,4,5,6,7,8",double train_size_split = 0.7,bool isdebug=true);
                          
                           double  LogisticRegressionMain(double& accuracy); 
                           void    ConfusionMatrix(double &y[], int &Predicted_y[], double& accuracy,bool print=true);

  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::CLogisticRegression(void)
 {
    e = 2.718281828; 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CLogisticRegression::~CLogisticRegression(void)
 {
   FileClose(m_handle); 
    
   ArrayFree(m_AllDataArrayDouble);
   ArrayFree(ColumnsToEncode);
   ArrayFree(ColumnsToFix);
   ArrayFree(MembersToEncode); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::FixMissingValues(string columns)
 { 
   ushort separator = StringGetCharacter(",",0);
   StringSplit(columns,separator,ColumnsToFix); //Store the columns numbers to this Array   
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::LabelEncoder(string columns,string members)
 {
   ushort separator = StringGetCharacter(",",0);
   StringSplit(columns,separator,ColumnsToEncode); //Store the columns numbers to this Array   
   
   separator = StringGetCharacter(",",0);
   StringSplit(members,separator,MembersToEncode); //Store the columns numbers to this Array 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::FixMissingValuesFunction(double& Arr[],int column_number, int index)
 {     
   // the index argument is for getting the name of the column
       for (int i=0; i<ArraySize(ColumnsToFix); i++) 
           if ((int)ColumnsToFix[i]!=column_number) continue; //the column to fix has to match the column on the input
           else
             { 
               Print("FixMissingValues Function triggered ","\n starting to Fix Missing Values in ",DataColumnNames[index]!=""?DataColumnNames[index]: (string)column_number," Column ");
                //from  the function >>> void CLogisticRegression::FixMissingValues(double &Arr[])
                    int counter=0; double mean=0, total=0;
                     for (int j=0; j<ArraySize(Arr); j++) //first step is to find the mean of the non zero values
                        if (Arr[j]!=0)
                          {
                            counter++;
                            total += Arr[j];
                          }
                         
                        mean = total/counter; //all the values divided by their total number
                       
                       if (m_debug)
                        {
                          Print("mean ",MathRound(mean)," before Arr");
                          ArrayPrint(Arr);
                        }
                        
                       for (int j=0; j<ArraySize(Arr); j++)
                         {
                           if (Arr[j]==0)
                             {
                               Arr[j] = MathRound(mean); //replace zero values in array
                             }
                         }
                      
                      if (m_debug)
                       {
                          Print("After Arr");
                          ArrayPrint(Arr); 
                       }
             } 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::EncodeLabelFunction(string& Arr[], double& output_arr[], int column_number,int index) //since the input is a string The Arr[] has to be a string
 { 
   // the index argument is for getting the name of the column
     double EncodeTo[];
     ArrayResize(EncodeTo,ArraySize(Arr)); //Make the Encode To Array Same size as source
     
     for (int i=0; i<ArraySize(ColumnsToEncode); i++) 
        if ((int)ColumnsToEncode[i]!=column_number) continue; //the column to fix has to matches the column on the input
        else
          {  
           Print("Encode Label Function triggered ","\n starting to Encode Values in ",DataColumnNames[index]!="" ? DataColumnNames[index]: (string)column_number," Column ");
           //from >>>> void CLogisticRegression::LabelEncoder(string &src[],int &EncodeTo[],string members="male,female")
            int binary=0;
            for(int x=0; x<ArraySize(MembersToEncode); x++) // loop the members array
              {
                 string val = MembersToEncode[x];
                 binary = x; //binary to assign to a member
                 int label_counter = 0;
                 
                 for (int y=0; y<ArraySize(Arr); y++) //source array
                    {
                      string source_val = Arr[y];
                       if (val == source_val)
                         {
                          EncodeTo[y] = binary;
                          label_counter++;
                         }
                    } 
                 Print(MembersToEncode[binary]," total =",label_counter," Encoded To = ",binary);
                 ArrayCopy(output_arr,EncodeTo);
              } 
          }
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CLogisticRegression::Init(string filename=NULL,string delimiter=",",int y_column=1,string x_columns="3,4,5,6,7,8",double train_size_split = 0.7,bool isdebug=true)
 {
//---
   LrInit(y_column,x_columns,filename,delimiter,train_size_split,false);
 
   
   LinearRegMain();
//---

   single_rowtotal = rows_total/x_columns_chosen;
 }
//+------------------------------------------------------------------+
//|       The Main Multiple regression Function for scripts          |
//+------------------------------------------------------------------+ 
double CLogisticRegression::LogisticRegressionMain(double &accuracy)
 {
   each_arrsize = (int) MathFloor(single_rowtotal*(1-m_train_split)); 
   
   int TestPredicted[];
   ArrayResize(TestPredicted,each_arrsize); 
    
    for (int i=0; i<each_arrsize; i++)
       {          
          double sigmoid = 1/(1+MathPow(e,-TestYDataSet[i])); 
          TestPredicted[i] = (int) round(sigmoid); //round the values to give us the actual 0 or 1
       }
    
//---

    ConfusionMatrix(TestYDataSet,TestPredicted,accuracy);
    
    string dates[];
    GetColumnDatatoArray(1,dates);
    string Temp_dates[];
    ArrayResize(Temp_dates,each_arrsize);
    
    int counter = 0;

    for (int i=0; i<ArraySize(dates); i++)
        if (i >= (int) MathCeil(single_rowtotal*m_train_split))
              Temp_dates[counter++] = dates[i];

//---

    ArrayFree(dates);
    ArrayCopy(dates,Temp_dates);
    ArrayFree(Temp_dates);
     
    
//---

    return (accuracy); 
 }
//+------------------------------------------------------------------+
//|       The Main Multiple regression Function for EA's             |
//+------------------------------------------------------------------+ 
void CLogisticRegression::ConfusionMatrix(double &y[], int &Predicted_y[], double& accuracy,bool print=true)
 {
    int TP=0, TN=0,  FP=0, FN=0; 
    
    for (int i=0; i<ArraySize(y); i++)
       {
         if ((int)y[i]==Predicted_y[i] && Predicted_y[i]==1)
            TP++;
         if ((int)y[i]==Predicted_y[i] && Predicted_y[i]==0)
            TN++;
         if (Predicted_y[i]==1 && (int)y[i]==0)
            FP++;
         if (Predicted_y[i]==0 && (int)y[i]==1)
            FN++;
       }
      
      if (print) 
       Print("Confusion Matrix \n ","[ ",TN,"  ",FP," ]","\n","  [  ",FN,"  ",TP,"  ] ");
     
     accuracy = (double)(TN+TP) / (double)(TP+TN+FP+FN);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
#define VAR_NAME(v) ""+#v+""

void CLogisticRegression::WriteToCSV(int &Predicted[],string &date_time[],string file_name,string delimiter=",")
 {
   FileDelete(file_name);
   
   int handle  = FileOpen(file_name,FILE_WRITE|FILE_READ|FILE_CSV|FILE_ANSI,delimiter); 

    if (handle == INVALID_HANDLE)
         Print(__FUNCTION__," Invalid csv handle err=",GetLastError());
         
//---
     if (handle>0)
       {  
         FileWrite(handle,VAR_NAME(Predicted),VAR_NAME(date_time)); 
            for (int i=0; i<ArraySize(Predicted); i++)
              {  
                string str1 = IntegerToString(Predicted[i]),
                str2 = date_time[i];
                FileWrite(handle,str1,str2); 
              }
       }
     FileClose(handle); 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
