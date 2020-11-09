# Optimizing an ML Pipeline in Azure

## Overview
**This project is part of the Udacity Azure ML Nanodegree.**

Where we start by building and optimizing an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. Once finished we compare the accuracy of the best model to an Azure AutoML run. 

## Summary
The Goal of this project is to predict if a client will subscribe to a terme deposit product (Variable value: ('yes') ,('no')) by using a dataset  (located  : https://www.kaggle.com/henriqueyamahata/bank-marketing ) related to direct marketing campaigns of a Portuguese banking institution.

**The main step of the project are :**

1. Create a model using Azure Python SDK and a LogisticRegression model
2. Tune the parameter of the Scikit-learn model using hyperdrive 
3. Create a Model using Azure Auto ML 
4. Compare the best performing model between the two created solutions based on their Accuracy 

**The best model is  :**

The best performing model **Using the VotingEnsemble Algorithm** with the Accuracy of **0.92014** was the result of the **Auto ML run.**

![Auto ML Run](a.PNG "Auto ML Run")


## Scikit-learn Pipeline :

The process of the created Pipeline is composed of two principal Step : 

  A. Create the Model using Python : (**train.py**)

1. Create the TabularDataset by using TabularDatasetFactory
2. Clean the data by using One hote encoding technique to deal with the discret features 
3. Split the data into training and testing sets.
4. Create the model using the LogisticRegression model.
5. Calculat the model Accuracy

  B. Tune the model Parameters using Hyperdrive  : (**udacity-project.ipynb**)
The parameters are **(C : "Inverse of regularization strength" , max_iter : "Maximum number of iterations to converge")**

1. Define the parameter sampling method to use over the hyperparameter space where we specify a liste of discret value used during the tuning *this choice of value was made after multiple execution of the hyperdrive run*
2. Specify the early stopping policy to Automatically terminate poorly performing runs every time the training script reports the primary metric
4. Create the SKLearn estimator 
5. Define the hyperdrive configuration , submit the run and register the best model **(C = 0.01 , max_iter = 400 give an Accuracy of 0.913)** using the result of the parameter tunning 

![Hyperdrive run](b.PNG "Hyperdrive run")
![Hyperdrive metric](c.PNG "Hyperdrive metric")
![Hyperdrive model registry](d.PNG "Hyperdrive model registry")


## AutoML

The process of the created solution is composed of two principal Step : 

  A. Define the Tabular dataset ,clean  the data and specify the training the the testing sets  (**like the Scikit-learn Pipeline**)
  A. Specify the configuration for the auto ML run so that he can choose the best model and hyperparameters for us
  ![auto ml configuration](e.png "auto ml configuration")

  
  



## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
