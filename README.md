# ML-Research-Hackathon
Predict wildfires with long containment time to help with evacuation policy, using Machine Learning and Explainable AI.

## Data
- The wildfire data is from https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires (not attached because of size limit). 
- Climate data can be found at files starting with 'climdiv'. The source is https://www.ncei.noaa.gov/cdo-web/datasets.

## Feature Engineering
- "CreateFeature.py" is the script for creating the climate feature and the historic containment time.
- Added columns are:
  - 'tmp_monthly': the average temprature of the month in the state when and where the fire happened.
  - 'pcp_monthly': the average precipitition of the month in the state when and where the fire happened.
  - 'pdsi_monthly': the Palmer Drought Severity Index of the month in the state when and where the fire happened.
  - 'NEARBY_HOUR_TO_CONT': the average containment time (in hours) of fires in the nearby areas in the past half year.

## Data Description and Visualization
- Code for generating all descriptive statistics and figures in the report can be found in "RH Wildfires Visualization.ipynb". 

## Model Building and Evaluation
- See "RH Wildfires ML Classification.ipynb" for the ML model building, and the explanary AI implementation.
- See the "State Analysis" section in "RH Wildfires Visualization.ipynb" for the zoom-in analysis process.
