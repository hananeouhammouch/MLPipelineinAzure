# Optimizing an ML Pipeline in Azure

## Overview
**This project is part of the Udacity Azure ML Nanodegree.**

Where we start by building and optimizing an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. Once finished we compare the accuracy of the best model to an Azure AutoML run. 

## Summary of the project <a name="Summary"></a>
The purpose of this project is to predict if a client will subscribe to a term deposit product by using a dataset  (located here: https://www.kaggle.com/henriqueyamahata/bank-marketing ) related to direct marketing campaigns of a Portuguese banking institution by using Azure ML pipeline (Python SDK) and Azure AutoML in two different process.

![projectdiagrame](projectdiagrame.png "projectdiagrame")

Once the two experiments over and the best models generated we compared their performance using the **Accuracy** as a primary metric, which leads us to conclude that the best solution resulting from the **Auto ML run** is a model based on the **VotingEnsemble Algorithm** because he give us an **Accuracy of 0.92014**

# Table of contents
1. [Scikit-learn Pipeline](#Scikit)
    1. [Create the Model using Python](#subparagraph1)
    2. [Tune the Parameters using Hyperdrive](#subparagraph2)
    3. [Result](#subparagraph3)
2. [AutoML Pipeline](#AutoML)
    1. [Configuration](#subparagraph11)
    2. [Prepocicing](#subparagraph12)
    3. [Result](#subparagraph13)
3. [Pipeline comparison](#comparison)
4. [Future work](#Future)

## Description of the Scikit-learn Pipeline :<a name="Scikit"></a>

You can find below a small description of the model creation and the choices made during this first step of the project :

  **A. Create the Model using Python :** (**train.py**) <a name="subparagraph1"></a>

We start  first by creating a tabular dataset **TabularDatasetFactory** from the the datasource to do some exploration and understind the meaning of each features once than we start to prepare and clean the data by using one hot encodig technique to deal with discrete features , after the preparation we split the result to training and testing sets. 

We then move to choising the best algorithm for our classification problem, which is **LogisticRegression** because we are trying  to predict if a client will subscribe to a term deposit product (yes or no) which mean **two (and only two) outcomes** . After the creation of the model we calculate it's **Accuracy**

Choosing the model based on only two parameters and after one run does not ensure that it will be functional in production which move us to the second step of this pipeline **Tune the model Parameters using Hyperdrive**

 **B. Tune the model Parameters using Hyperdrive  :** (**udacity-project.ipynb**) <a name="subparagraph2"></a>
  
To improve the Accuracy of our model we optimize our hyperparameters using Azure Machine Learning's tuning capabilities **Hyperdrive**.

First of all  , we define the hyperparameter space to sweep over. which mean tuning the **C** and **max_iter** parameters. In this step we use random sampling **RandomParameterSampling** to try different configuration sets of hyperparameters to maximize our primary metric, Accuracy.
*We use RandomParameterSampling which Defines random sampling over a hyperparameter search space to sample from a set of discrete values for max_iter and C hyperparameters. This will make the hyperparameter tunning choice more specific*

We then define our termination Policy for every run using **BanditPolicy** based on a slack factor equal to 0.01 as criteria for evaluation to  conserves resources by terminating runs that are poorly performing.
*This choice means that the primary metric of every run Y using this formula (Y + Y * 0.01) will be compared to the best metric of the hyperdrive execution, and if smaller, it cancels the run. this will assure that every run will give better accuracy than the one before*

Once than we create SKLearn estimator , Define the hyperdrive configuration and finally, lauch the hyperparameter tuning job.







![Hyperdrive run](b.PNG "Hyperdrive run")
![Hyperdrive metric](c.PNG "Hyperdrive metric")
![Hyperdrive model registry](d.PNG "Hyperdrive model registry")


## AutoML <a name="AutoML"></a>
<a name="subparagraph11"></a>
<a name="subparagraph12"></a>
<a name="subparagraph13"></a>


The process of the created solution is composed of three principal Step : 

  A. Define the Tabular dataset, clean  the data and specify the training and the testing sets  (**similar to the Scikit-learn Pipeline**)
  
  B. Specify the configuration for the auto ML run (**udacity-project.ipynb**)
  
  ![auto ml configuration](e.png "auto ml configuration")
  
  C. Submit the run and register the best model **(VotingEnsemble Algorithm and the Accuracy is 0.92014)** which involves summing the predictions made by multiple other classification models.
   
 ![automl model registry](g.PNG "automl model registry")
 
 *#More detail: Below the top algorithm tested by Auto ML Orderd by their Accuracy*

![automl result](f.PNG "automl result")
  
  
## Pipeline comparison <a name="comparison"></a>

To analyze the distinction among the two models we used the Accuracy as a primary metric and the outcome was that Auto ML provides more high-grade performance.
This result is coherent mostly because Auto ML run not only test more hyperparameter value than the Scikit-learn process but also more algorithm too 

## Future work <a name="Future"></a>

The improvement can be made not only in the Auto ml process by not using the cleaned data function (train.py) and leave the featurization to the Auto ML run **(to handle the Imbalanced data)** . but also in the Scikit-learn process by using other algorithm and testing other configuration to tune the hyperparameter

## Proof of cluster clean up <a name="delete"></a>

Once finished we delete the compute instance and the compute cluster used during this project to not incur any charges.

 ![Compute cluster delete](h.PNG "Compute cluster delete")
 ![Compute instance delete](i.PNG "Compute instance delete")

