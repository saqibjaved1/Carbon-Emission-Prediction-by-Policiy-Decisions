#Deployment

###Build from the source:

1: Setup virtual environment.

```python3 -m venv ./venv```  
```source venv/bin/activate```

2. Install dependencies

```pip install -r requirements.txt```

3. Start the web-app

```python src/app.py```

Server is then available at http://127.0.0.1:8050

###Deploy Docker container for web-app

If docker is not installed follow [this](https://docs.docker.com/engine/install/) link.

1. Execute startup script.

```bash start.sh```

The script first checks if docker image is locally available. If not, image is 
pulled from the docker hub.

If any error is encountered during docker deployment, follow the following steps:

1. Fetch docker image for the project: ```docker pull t6nand/group07-co2```

2. Deploy container: `docker run -d -p 8050:8050 t6nand/group07-co2`


# Web Application Requirement

Requires Python 3.5+

## Start server: 
```python src/app.py``` 

## Open web app:
```http://127.0.0.1:8050/```


## Run Data Analysis Pipeline
To run evaluation on a pretrained model for some test data set, 
the following script can be run:

```python tools/evaluate_data_analysis_pipeline.py --country "Some Country"```

If positional parameter `--country` is not provided, the default setting 
picks up "Germany" for evaluation.

The data analysis part can be understood in following 3 steps:
1. Data Processing: Based on country provided, training data is fetched and 
split into train & test parts. Test part can then be used as previously 
unseen data for testing on some pre-trained model. This pipeline provides 
multiple options catering to different models like choice of selecting 
specific government policies as features, normalize/standardize data or not, 
fill NaNs if requested, etc. 
2. Data Modeling: To avoid retraining for this part, one of the previously trained 
models can be considered. Candidate models can be found in the `content/checkpoint`
folder. In this script we have used CNN2, LSTM2, DNN, and NAIVE FORECAST for 
this purpose (Naming conventions of models as defined in the Milestone 3 
report). Overall 10 models were considered as documented in the milestone 3 
report.
3. Evaluation: In this part, test data fetched from step 1 can be used on the 
pre-trained model for making prediction and fetch performance metrics i.e. MSE, 
MAE and Soft Accuracy scores as also defined in the Milestone 3 report.