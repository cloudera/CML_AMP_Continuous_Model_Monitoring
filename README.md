# Continuous Model Monitoring

A demonstration of how to perform continuous model monitoring on Cloudera Machine Learning (CML) using the [Model Metrics](https://docs.cloudera.com/machine-learning/cloud/model-metrics/topics/ml-enabling-model-metrics.html) feature and [Evidently.ai's](https://evidentlyai.com/) open-source monitoring dashboards.

![](data/images/evidently_ai_logo_fi.png)

After iterations of development and testing, deploying a well-fit machine learning model often feels like the final hurdle for an eager data science team. In practice, however, a trained model is never final, and this milestone marks just the beginnning of a new chapter in the ML lifecycle called production ML. This is because most machine learning models are static, but the world we live in is dynamically changing all the time. Changes in environmental conditions like these are referred to as _concept drift,_ and will cause the predictive performance of a model to degrade over time, eventually making it obsolete for the task it was initially intended to solve.

> For an in-depth description of this problem space, please see our research report:
>
> [FF22: Inferring Concept Drift Without Labeled Data](https://concept-drift.fastforwardlabs.com/)

To combat concept drift in production systems, its important to have robust monitoring capabilities that alert stakeholders when relationships in the incoming data or model have changed. In this Applied Machine Learning Prototype (AMP), we demonstrate how this can be achieved on CML. Specifically, we leverage CML's [Model Metrics](https://docs.cloudera.com/machine-learning/cloud/model-metrics/topics/ml-enabling-model-metrics.html) feature in combination with Evidently.ai's [Data Drift](https://docs.evidentlyai.com/reports/data-drift), [Numerical Target Drift](https://docs.evidentlyai.com/reports/num-target-drift), and [Regression Performance](https://docs.evidentlyai.com/reports/reg-performance) reports to monitor a simulated production model that predicts [housing prices](https://www.kaggle.com/harlfoxem/housesalesprediction) over time.

## Project Structure

```
.
├── LICENSE
├── README.md
├── .project-metadata.yaml              # declarative specification for AMP logic
├── apps
│   ├── reports                         # folder to collect monitoring reports
│   └── app.py                          # Flask app to serve monitoring reports
├── cdsw-build.sh                       # build script for model endpoint
├── data                                # directory to hold raw and working data artifacts
├── requirements.txt
├── scripts
│   ├── install_dependencies.py         # commands to install python package dependencies
│   ├── predict.py                      # inference script that utilizes cdsw.model_metrics
│   ├── prepare_data.py                 # splits raw data into training and production sets
│   ├── simulate.py                     # script that runs simulated production logic
│   └── train.py                        # build and train an sklearn pipelne for regression
├── setup.py
└── src
    ├── __init__.py
    ├── api.py                          # utility class for working with CML APIv2
    ├── inference.py                    # utility class for concurrent model requests
    ├── simulation.py                   # utility class for simulation logic
    └── utils.py                        # various utility functions
```

By launching this AMP on CML, the following steps will be taken to recreate the project in your workspace:

1. A Python session is run to install all required project dependencies
2. A Python session is run to split the raw data into training and production sets, then saved locally
3. A [sci-kit learn pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) with preprocessing and ridge regression steps is constructed and used in a grid search - cross validation to select the best estimator among a set of hyperparameters. This pipeline is save to the project.
4. The pipeline is deployed as a [hosted REST API](https://docs.cloudera.com/machine-learning/cloud/models/topics/ml-models.html) with CML's [Model Metrics feature](https://docs.cloudera.com/machine-learning/cloud/model-metrics/topics/ml-enabling-model-metrics.html) enabled to track each prediction with a managed Postgres database for later analysis.
5. A [simulation](src/simulation.py) is run in that iterates over the production dataset in monthly batches. For each new month of production data (of which there are six total), the simulation will:
   - Lookup newly _listed_ properties from the batch and predict their sale prices using the deployed model
   - Lookup newly _sold_ properties from the batch and track their ground truth values by joining to original prediction record in the metric store
   - Calculate drift metrics and deploy a refreshed, Evidently monitoring dashboard via a [CML Application](https://docs.cloudera.com/machine-learning/cloud/applications/topics/ml-applications-c.html) 

Upon succesful recreation of the project (which may take ~20 minutes), the simulation will have produced six monitoring reports - one for each month of "production" records - and saved those reports to the `apps/reports/` directory. Each report consists of three Evidently report tabs (data drift, target drift, and regression performance) that are combined into a single application that you can access directly via the Applications pane in CML to determine if and where drift is occuring within the new batch of data. We encourage users to peruse the [simulation logic and documentation](src/simulation.py) directly for a detailed look at how new records are scored, logged, and queried to generate monitoring reports.

> NOTE: Since the simulation is intended to mimic a production scenario, the deployed application is refreshed *in-place* with results from each new batch of data. Therefore, only the most recent month's drift report is displayed at any given time. You can inspect the deployed application while the simulation is running see month-to-month changes in the drift reports.

## Launching the Project on CML

This AMP was developed against Python 3.6. There are two ways to launch the project on CML:

1. **From Prototype Catalog** - Navigate to the AMPs tab on a CML workspace, select the "Continuous Model Monitoring" tile, click "Launch as Project", click "Configure Project"
2. **As an AMP** - In a CML workspace, click "New Project", add a Project Name, select "AMPs" as the Initial Setup option, copy in this repo URL, click "Create Project", click "Configure Project"

