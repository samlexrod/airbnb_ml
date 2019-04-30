# Airbnb_ML

## Project Motivation
The motivation to do this project stated with the suggested datasets to work with provided by Udacity. Learning about the CRISP-DM methodology sparked the idea of using it to disect business knowledge. Sometimes it is difficult to start a project due to lack of knowledge in the field or business. However, the steps of it made it easier to como up with useful questions along the way of the process. 

## Project Info
* This projects focuses on the prediction of listing prices over a date period. 
* To predict the prices, a linear regression and a random forest regression was applied. 
* The business questions formulated were answered with visualizations using either seaborn or matplotlib.

The data comes from Kaggle.com. And the dataset analyzed is for [Seattle](https://www.kaggle.com/airbnb/seattle/data). 

The article corresponding to the Python code was wirtten on Medium.com.
https://medium.com/@samlexrod/crisp-dm-regression-analysis-in-python-ade4b74bdc85?source=friends_link&sk=3629b7e62926e5483b09311380fb1653

## Context
This datasets describes the listing activity of homestays in Boston, MA and Seattle, WA.

## Files and Folder Descriptions

data : It contains the Airbnb datasets
* Listings, including full descriptions and average review score 
* Reviews, including unique id for each reviewer and detailed comments 
* Calendar, including listing id and the price and availability for that day 

images : It contains the images of graphs and tables used in the [article](https://medium.com/@samlexrod/crisp-dm-regression-analysis-in-python-ade4b74bdc85).

.gitignore : this are the files not synced on GitHub. To know more about this file visit: https://www.git-scm.com/docs/gitignore

Airnb_Seattle_Complex.ipynb : It contains the complex analysis. This file does not follow the same order as the article since it contains more complex analyses.

Airbnb_Seattle_Simple.ipynb : It contains the simple analysis. This file does support the article on Medium.com. This is the file to use to reference what is being covered on ["Dive in the CRISP-DM to Understand and Predict Seattle Listing Prices"](https://medium.com/@samlexrod/crisp-dm-regression-analysis-in-python-ade4b74bdc85?source=friends_link&sk=3629b7e62926e5483b09311380fb1653)

LICENSE : This is the license to use the code. It basically says, "Use it however you like at your own risk."

README : You are reading me!

airbnb_ml.yml : This is the anaconda enviroment used for the analysis. The instructions on how to recreate this environmnet is going to be in the installation section below.

helper.py : This is the file that contains the modular code. It contains useful classes, functions, and methods designed specifically for the analysis to avoid repetition of code.

    
## Installation

Make sure the data is located inside a data folder named "data". 
There is no need to unzip the files since the helper file takes care of it.

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

We found many good insights about this data even though there is more to explore:

* One, cleaning is necessary; the dataset included unavailable listings and numeric characters included non-numeric characters.
* Two, not all months were provided in full.
Three, most months are listed with prices around $100.
* Four, $0 cleaning fees are provided as nulls (although this assumption could be validated by going back to the source of the data).
* Five, most property types have 1 bathroom.
* Six, the majority of listings are mainly either a house or an apartment; in combination, they account for 80% of the listings.
* Seven, Seattle listings prices are not significantly related to its property type.
* Eight, outliers or listings prices outside the norm are concentrated in two listings. One listing seems to be increasing its price. The other listing perhaps realized the listing was too expensive and lowered its listing to more normal prices.
* Nine, based on the selected features, we can use the model to suggest normal prices to new hosts or listings base on the description of each listing. We can do this with a ~69% (simple model) or a ~92% (complex model) explanation by the chosen features.
   

## Thank you for visiting my repository!
