# Airbnb_ML

This projects focuses on the prediction of listing prices over a date period. The data comes from Kaggle.com. And the dataset analyzed is for [Seattle](https://www.kaggle.com/airbnb/seattle/data). 

Make sure the data is located inside a data folder named "data". 
There is no need to unzip the files since the helper file takes care of it.

## Context
This datasets describes the listing activity of homestays in Boston, MA and Seattle, WA.

## Content
The following Airbnb activity is included in this Seattle dataset: 
    * Listings, including full descriptions and average review score 
    * Reviews, including unique id for each reviewer and detailed comments 
    * Calendar, including listing id and the price and availability for that day 
    
## Environment
The environment used for this analysis is the Anaconda distribution using Python 3. 

The libraries used are the following:
In jupyter notebook:
* pandas
* numpy
* seaborn
* matplotlib.pyplot
* os
* re
* sklearn.preprocessing.OneHotEncoder
* wordcloud
* zipfile.ZipFile

The airbnb_ml.yml file in this repository contains all the packages used for this analysis.

## Registering the Environment to Use in Jupyter

I exported the environment using (while activated):
```
conda env export > airbnb_ml.yml
```
FYI: to remove an environment use (while deactivated): 
```
conda remove --name airbnb_ml --all
```

To recreate the environment type the following in anaconda prompt:
```
conda env create -f airbnb_ml.yml

conda install -n airbnb_ml ipykernel

activate airbnb_ml

source activate jupyter-airbnb_ml

ipython kernel install --user --name jupyter-airbnb_ml

jupyter notebook
```

## Thank you for visiting my repository!
