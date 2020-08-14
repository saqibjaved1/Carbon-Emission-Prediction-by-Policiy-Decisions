This is our git repo!

# Development Section

## Create and activate environment
```python3 -m venv ./venv```  
```source venv/bin/activate```

## Install packages
```pip install -r requirements.txt```

## Install predictCO2 package
```pip install -e .```

## Run tests
```pytest ./tests```

## Run linter
```pylint src```

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
folder. In this script we have used CNN2 for this purpose. 
(Naming conventions of models as defined in the Milestone 3 report.)
3. Evaluation: In this part, test data fetched from step 1 can be used on the 
pre-trained model for making prediction and fetch performance metrics i.e. MSE, 
MAE and Soft Accuracy scores as also defined in the Milestone 3 report. 