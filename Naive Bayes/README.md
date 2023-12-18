There are three types of Naive bayes algorithms Under this class;

## Multinomial Naive bayes.
This is used for discrete counts, for example text classification the class CNaiveBayes is responsible for this

## Gaussian naive bayes

is used for classification too as it assumes that features are independent to each other and they follow 
the normal distribution
CGaussianNaiveBayes:: is the class responsible for this

## Bernoulli:

This binomial model is useful, if your feature vectors are binary ones, meaning the zeros and ones

A class for this won't be available in this library, as I think it is irrelevant to the trading scenarios
that this library was primarily built to perform

## Advantages of Naive Bayes classifier

It is one of the fastest ML algorithms to predict a class of dataset
It can be used for binary as well as multi-class classifiers
It performs well in multi-class predictions compared to other algorithms
It is the most popular choice for text classification problems

## Disadvantages of the Naive Bayes Classifier

if a categorical variable is in the new dataset, but wasn't observed in the training dataseet, the model will
assign it to have a probability of zero, since probabilities depend on the prior evidence
The naive bayes assumes that the features varibles are completely independent, this might prove to be wrong often times
but since this was built to make trading decisions, who cares ?? by the way who knows what causes the market to behave the way
it does and how are such thing as indicators relate to one another?


## Applications of the Naive Bayes classifier

It is used for credit scoring 
it is used in medical data classification
it is mostly used for text clssifcation problems such as spam filtering





