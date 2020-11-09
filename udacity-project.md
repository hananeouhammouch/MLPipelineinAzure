```python
from azureml.core import Workspace, Experiment

#Load the existing workspace , create the experiment and start the logging: 
ws = Workspace.get(name="quick-starts-ws-125428")
exp = Experiment(workspace=ws, name="quick-starts-ws-125428")

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

run = exp.start_logging()
```

    Workspace name: quick-starts-ws-125428
    Azure region: southcentralus
    Subscription id: 572f8abf-a1a0-4b78-8c6d-3630739c72b5
    Resource group: aml-quickstarts-125428



```python
from azureml.core.compute import ComputeTarget, AmlCompute

# Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

from azureml.core.compute_target import ComputeTargetException

amlcompute_cluster_name = "cpu-cluster"

try:
    aml_compute = ComputeTarget(workspace=ws, name=amlcompute_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    aml_compute = ComputeTarget.create(ws, amlcompute_cluster_name, compute_config)

aml_compute.wait_for_completion(show_output=True)
```

    Creating
    Succeeded
    AmlCompute wait for completion finished
    
    Minimum number of nodes requested have been provisioned



```python
from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import choice
import os

# Specify parameter sampler
ps = RandomParameterSampling( {
    "--C":  choice(0.01, 0.02, 0.03, 0.04, 0.05),
    "--max_iter":  choice(100, 200, 300, 400, 500)
    }
)
# Specify a Policy
policy =BanditPolicy(evaluation_interval=1, slack_factor=0.001)

if "training" not in os.listdir():
    os.mkdir("./training")

# Create a SKLearn estimator for use with train.py
est = SKLearn(source_directory='./', 
                entry_script='train.py', compute_target=aml_compute)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config =HyperDriveConfig(hyperparameter_sampling=ps, 
                                    primary_metric_name='Accuracy', 
                                    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                    policy=policy,
                                    max_total_runs=20,
                                    max_concurrent_runs=4,
                                    estimator=est
                                   )
```


```python
# Submit your hyperdrive run to the experiment and show run details with the widget.
hyperdrive_run = exp.submit(config=hyperdrive_config)

RunDetails(hyperdrive_run).show()

hyperdrive_run.wait_for_completion(show_output=True)

```

    WARNING - If 'script' has been provided here and a script file name has been specified in 'run_config', 'script' provided in ScriptRunConfig initialization will take precedence.



    _HyperDriveWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO'…




    RunId: HD_ffab1f70-ebb5-437c-a956-778b81ca77b9
    Web View: https://ml.azure.com/experiments/quick-starts-ws-125428/runs/HD_ffab1f70-ebb5-437c-a956-778b81ca77b9?wsid=/subscriptions/572f8abf-a1a0-4b78-8c6d-3630739c72b5/resourcegroups/aml-quickstarts-125428/workspaces/quick-starts-ws-125428
    
    Streaming azureml-logs/hyperdrive.txt
    =====================================
    
    "<START>[2020-11-09T18:27:47.057173][API][INFO]Experiment created<END>\n""<START>[2020-11-09T18:27:47.707826][GENERATOR][INFO]Trying to sample '4' jobs from the hyperparameter space<END>\n""<START>[2020-11-09T18:27:47.993713][GENERATOR][INFO]Successfully sampled '4' jobs, they will soon be submitted to the execution target.<END>\n"<START>[2020-11-09T18:27:48.5849262Z][SCHEDULER][INFO]The execution environment is being prepared. Please be patient as it can take a few minutes.<END>
    
    Execution Summary
    =================
    RunId: HD_ffab1f70-ebb5-437c-a956-778b81ca77b9
    Web View: https://ml.azure.com/experiments/quick-starts-ws-125428/runs/HD_ffab1f70-ebb5-437c-a956-778b81ca77b9?wsid=/subscriptions/572f8abf-a1a0-4b78-8c6d-3630739c72b5/resourcegroups/aml-quickstarts-125428/workspaces/quick-starts-ws-125428
    





    {'runId': 'HD_ffab1f70-ebb5-437c-a956-778b81ca77b9',
     'target': 'cpu-cluster',
     'status': 'Completed',
     'startTimeUtc': '2020-11-09T18:27:46.727067Z',
     'endTimeUtc': '2020-11-09T18:41:00.237136Z',
     'properties': {'primary_metric_config': '{"name": "Accuracy", "goal": "maximize"}',
      'resume_from': 'null',
      'runTemplate': 'HyperDrive',
      'azureml.runsource': 'hyperdrive',
      'platform': 'AML',
      'ContentSnapshotId': 'de0858ee-d1e4-4e0b-bfec-b7b15c5aacb1',
      'score': '0.9138088012139606',
      'best_child_run_id': 'HD_ffab1f70-ebb5-437c-a956-778b81ca77b9_11',
      'best_metric_status': 'Succeeded'},
     'inputDatasets': [],
     'outputDatasets': [],
     'logFiles': {'azureml-logs/hyperdrive.txt': 'https://mlstrg125428.blob.core.windows.net/azureml/ExperimentRun/dcid.HD_ffab1f70-ebb5-437c-a956-778b81ca77b9/azureml-logs/hyperdrive.txt?sv=2019-02-02&sr=b&sig=LxReMcG0jTKFRpYreI8Dc2serwyn5CKAjZIU1Vj%2FDd4%3D&st=2020-11-09T18%3A31%3A08Z&se=2020-11-10T02%3A41%3A08Z&sp=r'}}




```python
import joblib
# Get your best run and save the model from that run.
best_run = hyperdrive_run.get_best_run_by_primary_metric()
print(best_run.get_details()['runDefinition']['arguments'])
print(best_run.get_file_names())

model = best_run.register_model(model_name='Bank_marketing_model_hyperdrive', model_path='./')

```

    ['--C', '0.01', '--max_iter', '400']
    ['azureml-logs/55_azureml-execution-tvmps_200ef7d8e034fe18873821b7f19d5c959e0fab2145898b91464e050ea2275edb_d.txt', 'azureml-logs/65_job_prep-tvmps_200ef7d8e034fe18873821b7f19d5c959e0fab2145898b91464e050ea2275edb_d.txt', 'azureml-logs/70_driver_log.txt', 'azureml-logs/75_job_post-tvmps_200ef7d8e034fe18873821b7f19d5c959e0fab2145898b91464e050ea2275edb_d.txt', 'azureml-logs/process_info.json', 'azureml-logs/process_status.json', 'logs/azureml/108_azureml.log', 'logs/azureml/job_prep_azureml.log', 'logs/azureml/job_release_azureml.log']



```python
from azureml.data.dataset_factory import TabularDatasetFactory

# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

path_url = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv'
ds = TabularDatasetFactory.from_delimited_files(path = path_url)
```


```python
from train import clean_data
import pandas as pd

# Use the clean_data function to clean your data.
x, y = clean_data(ds)

# Create a dataframe from the cleaned data
df = pd.concat([x, y], axis = 1)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>marital</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>duration</th>
      <th>campaign</th>
      <th>pdays</th>
      <th>...</th>
      <th>contact_telephone</th>
      <th>education_basic.4y</th>
      <th>education_basic.6y</th>
      <th>education_basic.9y</th>
      <th>education_high.school</th>
      <th>education_illiterate</th>
      <th>education_professional.course</th>
      <th>education_university.degree</th>
      <th>education_unknown</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>57</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>371</td>
      <td>1</td>
      <td>999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>4</td>
      <td>285</td>
      <td>2</td>
      <td>999</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>52</td>
      <td>1</td>
      <td>999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>5</td>
      <td>355</td>
      <td>4</td>
      <td>999</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>5</td>
      <td>189</td>
      <td>2</td>
      <td>999</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split


#Split data into training and test and create a csv files with the result 
df_train, df_test = train_test_split(df, test_size=0.35)
df_train.to_csv("training/training_dataset.csv")
df_test.to_csv("validation/validation_dataset.csv")

```


```python
#Create an experiment in the default workspace
experiment = Experiment(ws, "auto_ml_BM_exp")

#Get the default datastore for the workspace.
datastore = ws.get_default_datastore()
```


```python
#Upload the training dataset and the validation dataset to the datastore 
datastore.upload(src_dir = "training/", target_path = "data/")
datastore.upload(src_dir = "validation/", target_path = "data/")

```

    Uploading an estimated of 1 files
    Uploading training/training_dataset.csv
    Uploaded training/training_dataset.csv, 1 files out of an estimated total of 1
    Uploaded 1 files
    Uploading an estimated of 1 files
    Uploading validation/validation_dataset.csv
    Uploaded validation/validation_dataset.csv, 1 files out of an estimated total of 1
    Uploaded 1 files





    $AZUREML_DATAREFERENCE_132c03f69c1448ac9b37d60f5075d480




```python
# Upload the training data and the validation data as a tabular dataset 
training_data = TabularDatasetFactory.from_delimited_files(path = [(datastore, ("data/training_dataset.csv"))])
validation_data = TabularDatasetFactory.from_delimited_files(path = [(datastore, ("data/validation_dataset.csv"))])
```


```python
from azureml.train.automl import AutoMLConfig
# Set parameters for AutoMLConfig
# NOTE: DO NOT CHANGE THE experiment_timeout_minutes PARAMETER OR YOUR INSTANCE WILL TIME OUT.
# If you wish to run the experiment longer, you will need to run this notebook in your own
# Azure tenant, which will incur personal costs.

automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric="accuracy",
    compute_target=aml_compute,
    training_data=training_data,
    validation_data =validation_data,
    label_column_name="y")

```


```python

# Submit your automl run
auto_ml_run = experiment.submit(config = automl_config, show_output = True)
RunDetails(auto_ml_run).show()

```

    Running on remote.
    Running on remote compute: cpu-cluster
    Parent Run ID: AutoML_46955e9e-3512-4325-a4c1-635fa16e3043
    
    Current status: FeaturesGeneration. Generating features for the dataset.
    Current status: DatasetBalancing. Performing class balancing sweeping
    Current status: ModelSelection. Beginning model selection.
    
    ****************************************************************************************************
    DATA GUARDRAILS: 
    
    TYPE:         Class balancing detection
    STATUS:       ALERTED
    DESCRIPTION:  To decrease model bias, please cancel the current run and fix balancing problem.
                  Learn more about imbalanced data: https://aka.ms/AutomatedMLImbalancedData
    DETAILS:      Imbalanced data can lead to a falsely perceived positive effect of a model's accuracy because the input data has bias towards one class.
    +---------------------------------+---------------------------------+--------------------------------------+
    |Size of the smallest class       |Name/Label of the smallest class |Number of samples in the training data|
    +=================================+=================================+======================================+
    |2367                             |1                                |21417                                 |
    +---------------------------------+---------------------------------+--------------------------------------+
    
    ****************************************************************************************************
    
    TYPE:         Missing feature values imputation
    STATUS:       PASSED
    DESCRIPTION:  No feature missing values were detected in the training data.
                  Learn more about missing value imputation: https://aka.ms/AutomatedMLFeaturization
    
    ****************************************************************************************************
    
    TYPE:         High cardinality feature detection
    STATUS:       PASSED
    DESCRIPTION:  Your inputs were analyzed, and no high cardinality features were detected.
                  Learn more about high cardinality feature handling: https://aka.ms/AutomatedMLFeaturization
    
    ****************************************************************************************************
    
    ****************************************************************************************************
    ITERATION: The iteration being evaluated.
    PIPELINE: A summary description of the pipeline being evaluated.
    DURATION: Time taken for the current iteration.
    METRIC: The result of computing score on the fitted pipeline.
    BEST: The best observed score thus far.
    ****************************************************************************************************
    
     ITERATION   PIPELINE                                       DURATION      METRIC      BEST
             0   MaxAbsScaler LightGBM                          0:00:34       0.9168    0.9168
             1   MaxAbsScaler XGBoostClassifier                 0:00:30       0.9162    0.9168
             2   MinMaxScaler RandomForest                      0:00:33       0.8970    0.9168
             3   StandardScalerWrapper SGD                      0:00:25       0.9108    0.9168
             4   MinMaxScaler RandomForest                      0:00:33       0.8851    0.9168
             5   StandardScalerWrapper SGD                      0:00:32       0.8589    0.9168
             6   StandardScalerWrapper RandomForest             0:00:29       0.9019    0.9168
             7   RobustScaler ExtremeRandomTrees                0:00:34       0.8932    0.9168
             8   StandardScalerWrapper ExtremeRandomTrees       0:00:37       0.8170    0.9168
             9   StandardScalerWrapper SGD                      0:00:36       0.9036    0.9168
            10   StandardScalerWrapper SGD                      0:00:30       0.9021    0.9168
            11   MinMaxScaler SGD                               0:00:32       0.8421    0.9168
            12   RobustScaler ExtremeRandomTrees                0:00:31       0.7292    0.9168
            13   MinMaxScaler SGD                               0:00:34       0.9110    0.9168
            14   MinMaxScaler ExtremeRandomTrees                0:00:33       0.8992    0.9168
            15   MinMaxScaler ExtremeRandomTrees                0:00:25       0.9012    0.9168
            16   MinMaxScaler ExtremeRandomTrees                0:00:37       0.8994    0.9168
            17   StandardScalerWrapper RandomForest             0:00:40       0.8153    0.9168
            18   StandardScalerWrapper SGD                      0:00:37       0.8418    0.9168
            19   RobustScaler ExtremeRandomTrees                0:00:36       0.8421    0.9168
            20   StandardScalerWrapper RandomForest             0:00:32       0.8253    0.9168
            21   MinMaxScaler ExtremeRandomTrees                0:00:28       0.7420    0.9168
            22   MaxAbsScaler LightGBM                          0:00:30       0.8926    0.9168
            23   MinMaxScaler RandomForest                      0:00:32       0.9026    0.9168
            24   MaxAbsScaler ExtremeRandomTrees                0:00:32       0.9012    0.9168
            25   StandardScalerWrapper LightGBM                 0:00:38       0.8851    0.9168
            26   MaxAbsScaler ExtremeRandomTrees                0:00:34       0.8960    0.9168
            27   MinMaxScaler SVM                               0:02:11       0.9031    0.9168
            28   SparseNormalizer LightGBM                      0:00:31       0.9087    0.9168
            29   MinMaxScaler ExtremeRandomTrees                0:00:34       0.8985    0.9168
            30   MinMaxScaler LightGBM                          0:00:33       0.9097    0.9168
            31   MinMaxScaler RandomForest                      0:00:29       0.8992    0.9168
            32    VotingEnsemble                                0:00:44       0.9201    0.9201
            33    StackEnsemble                                 0:00:55       0.9162    0.9201



    _AutoMLWidget(widget_settings={'childWidgetDisplay': 'popup', 'send_telemetry': False, 'log_level': 'INFO', 's…





```python
# Retrieve and save your best automl model.
best_run, fitted_model = auto_ml_run.get_output()
print(best_run)
print(fitted_model)

model_ml = best_run.register_model(model_name='Bank_marketing_model_auto_ml', model_path='./')
```

    Run(Experiment: auto_ml_BM_exp,
    Id: AutoML_46955e9e-3512-4325-a4c1-635fa16e3043_32,
    Type: azureml.scriptrun,
    Status: Completed)
    Pipeline(memory=None,
             steps=[('datatransformer',
                     DataTransformer(allow_chargram=None, enable_dnn=None,
                                     enable_feature_sweeping=None,
                                     feature_sweeping_config=None,
                                     feature_sweeping_timeout=None,
                                     featurization_config=None, force_text_dnn=None,
                                     is_cross_validation=None,
                                     is_onnx_compatible=None, logger=None,
                                     observer=None, task=None, working_dir=None)),
                    ('prefittedso...
                                                                                                      min_samples_split=0.10368421052631578,
                                                                                                      min_weight_fraction_leaf=0.0,
                                                                                                      n_estimators=50,
                                                                                                      n_jobs=1,
                                                                                                      oob_score=False,
                                                                                                      random_state=None,
                                                                                                      verbose=0,
                                                                                                      warm_start=False))],
                                                                         verbose=False))],
                                                   flatten_transform=None,
                                                   weights=[0.38461538461538464,
                                                            0.23076923076923078,
                                                            0.07692307692307693,
                                                            0.07692307692307693,
                                                            0.07692307692307693,
                                                            0.07692307692307693,
                                                            0.07692307692307693]))],
             verbose=False)



```python

```
