# Airbnb_ML

## Project Motivation
The motivation to do this project stated with the suggested datasets to work with provided by Udacity. Learning about the CRISP-DM methodology sparked the idea of using it to disect business knowledge. Sometimes it is difficult to start a project due to lack of knowledge in the field or business. However, the steps of it made it easier to como up with useful questions along the way of the process. 

## Project Info
* This projects focuses on the prediction of listing prices over a date period. 
* To predict the prices, a linear regression and a random forest regression was applied. 
* The business questions formulated were answered with visualizations using either seaborn or matplotlib.

The data comes from Kaggle.com. And the dataset analyzed is for [Seattle](https://www.kaggle.com/airbnb/seattle/data). 

Make sure the data is located inside a data folder named "data". 
There is no need to unzip the files since the helper file takes care of it.

## Context
This datasets describes the listing activity of homestays in Boston, MA and Seattle, WA.

## File Descriptions
The following Airbnb activity is included in this Seattle dataset: 
* Listings, including full descriptions and average review score 
* Reviews, including unique id for each reviewer and detailed comments 
* Calendar, including listing id and the price and availability for that day 


    
## Installation

### Anaconda
The environment used for this analysis is the Anaconda distribution using Python 3. 
To install Anaconda follow the instructions provided in the Anaconda website:
   https://docs.anaconda.com/anaconda/install/windows/

The [airbnb_ml.yml](https://github.com/sammyrod/Airbnb_ML/blob/master/airbnb_ml.yml) file in this repository contains all the packages used for this analysis.

### Registering the Environment to Use in Jupyter

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

ipython kernel install --user --name airbnb_ml

jupyter notebook
```

## Libraries Used

These are the main packages used:
   pandas
   numpy
   os
   matplotlib
   seaborn
   sklearn
   statsmodels
   warnings
   zipfile 
   re
   wordcloud 
   
## Results   
   

## Thank you for visiting my repository!
