# Optimizing an ML Pipeline in Azure

## Overview
**This project is part of the Udacity Azure ML Nanodegree.**

Where we start by building and optimizing an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. Once finished we compare the accuracy of the best model to an Azure AutoML run. 

## Summary
The purpose of this project is to predict if a client will subscribe to a term deposit product by using a dataset  (located here: https://www.kaggle.com/henriqueyamahata/bank-marketing ) related to direct marketing campaigns of a Portuguese banking institution.

**The main steps of the project are:**

1. Build a model using Azure Python SDK and a LogisticRegression algorithm
2. Tune the parameter of the Scikit-learn model using hyperdrive 
3. Design a Model using Azure Auto ML 
4. Compare the best performing model between the two created solutions based on their Accuracy 

**The best model:**

The best performing model **Using the VotingEnsemble Algorithm** with the Accuracy of **0.92014** was the result of the **Auto ML run.**

![Auto ML Run](a.PNG "Auto ML Run")

## Scikit-learn Pipeline :

The design of the created Pipeline is composed of two principal Step : 

  A. Create the Model using Python : (**train.py**)

1. Create the TabularDataset using TabularDatasetFactory
2. Clean the data using One hot encoding technique to deal with the discrete features 
3. Split the data into training and testing sets.
4. Build the model using the LogisticRegression algorithm.
5. Calculate the model Accuracy

  B. Tune the model Parameters using Hyperdrive  : (**udacity-project.ipynb**)

1. Define the parameter sampling method where we specify a list of discrete value to use during the tuning *this choice of value was made after multiple executions of the hyperdrive run*
2. Specify the early stopping policy to Automatically terminate poorly performing runs every time the training script reports the primary metric
4. Create the SKLearn estimator 
5. Define the hyperdrive configuration, submit the run and register the best model by using the result of the parameter tunning  **(C = 0.01 , max_iter = 400 give an Accuracy of 0.913)** 

*We use RandomParameterSampling which Defines random sampling over a hyperparameter search space to sample from a set of discrete values for max_iter and C hyperparameters . This will make the hyperparameter tunning choice more specific.

*We also use BanditPolicy which defines an early termination policy based on slack factore equal to 0.01 as a criteria for evaluation. This choice mean that the primary metric of every run Y using this formula (Y + Y * 0.01) will be compared to the best metric of the hyperdrive execution , and if smaller, it cancels the run. this will assure that every run will give better accuracy than the one before*

![Hyperdrive run](b.PNG "Hyperdrive run")
![Hyperdrive metric](c.PNG "Hyperdrive metric")
![Hyperdrive model registry](d.PNG "Hyperdrive model registry")


## AutoML

The process of the created solution is composed of three principal Step : 

  A. Define the Tabular dataset, clean  the data and specify the training and the testing sets  (**similar to the Scikit-learn Pipeline**)
  
  B. Specify the configuration for the auto ML run (**udacity-project.ipynb**)
  
  ![auto ml configuration](e.png "auto ml configuration")
  
  C. Submit the run and register the best model **(VotingEnsemble Algorithm and the Accuracy is 0.92014)**  
  
 ![automl result](f.PNG "automl result")
 ![automl model registry](g.PNG "automl model registry")
  
  
## Pipeline comparison

To analyze the distinction among the two models we used the Accuracy as a primary metric and the outcome was that Auto ML provides more high-grade performance.
This result is coherent mostly because Auto ML run not only test more hyperparameter value than the Scikit-learn process but also more algorithm too 

## Future work

The improvement can be made not only in the Auto ml process by not using the cleaned data function (train.py) and leave the featurization to the Auto ML run **(to handle the Imbalanced data)** . but also in the Scikit-learn process by using other algorithm and testing other configuration to tune the hyperparameter

## Proof of cluster clean up

Once finished we delete the compute instance and the compute cluster used during this project to not incur any charges.

 ![Compute cluster delete](h.PNG "Compute cluster delete")
 ![Compute instance delete](i.PNG "Compute instance delete")

