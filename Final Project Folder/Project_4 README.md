# Project 4
# Group name: House of Cards

# Group members:
Sridevi Madduri
Kokila Janarthanan
Huntley Boden
David Mostacero
Nicoleta Cosereanu

# Project Proposal: 
Create a model to predict election winners using various machine learning techniques

Intro: The relationship between demographic factors and election results is complex and multifaceted. While population size can be an important factor in determining election outcomes, it is not the only factor that matters. Other demographic factors such as income, age, and poverty rate can also play a role in shaping voter preferences. 

For example, areas with higher household income may tend to vote for candidates who advocate for policies that benefit the wealthy, while areas with lower household income may be more likely to support candidates who prioritize policies that address economic inequality. Similarly, areas with a higher average age may be more likely to vote for candidates who focus on issues such as retirement benefits and healthcare for seniors, while areas with a younger population may be more concerned with issues such as education and job opportunities.

Overall, understanding the relationship between demographic factors and election results can be an important tool for election campaign managers in developing strategies to target specific voter groups and appeal to their priorities and concerns.

# Modeling approach:
Because we are trying to classify the election reuslts (Republican or Democrate) this project focuses exclusively on supervised machine learning techniques 

# <span style="color: blue">Instructions:</span>

Run <span style="color: green"> election_ETL.ipynb </span> file firts<br />
Check you have the below dependencies:<br />

%matplotlib inline <br />
from matplotlib import pyplot as plt<br />
from sklearn.datasets import make_classification<br />
import numpy as np<br />
import pandas as pd<br />
from sklearn.ensemble import RandomForestClassifier<br />
from sklearn.linear_model import LogisticRegression<br />
from sklearn.model_selection import train_test_split<br />
from sklearn.preprocessing import StandardScaler<br />
import pymongo<br />
import requests<br />
from sklearn.metrics import confusion_matrix<br />
import seaborn as sns<br />
from sklearn.metrics import roc_curve, roc_auc_score<br />


# KNN Model

## Prepare the Data :
* Run the  election_ETL Jupyter notebook to extract , transform and load data into MongoDB.
* Import the data from MongoDB.
* Used PyMongo to work with MongoDB.
* Removed the "Winners" column from the dataset.
* Standardize the dataset (using StandardScaler) so that columns that contain larger values do 
  not influence the outcome more than columns with smaller values.

## Build KNN Model :
* To do KNN Modeling , We need target and features column.
* Make the target column as a binary indicator.   
* Select the columns that is going to have an effect in predicting the winner of the county elections as the features.
* Pick data for years 2012 and 2016, for training a model.
* Split them into training and testing data.
* Create a KNN Model.
* Fit(train) our model by using the training data.
* Show the model accuracy for training and test data.

## Predict year 2020 results using the Model:
* Showed the Actual and Predicted values of the winner and their differences.

# Support Vector Machin (SVM) Model
The SVM can be briefly described as an algorithm developed to find a hyperplane in an N-dimensional space where N equals the number of different features in a data set. 

For this project, the data was retrieved from the census_DB database stored in MongoDB 
Preprocessing consisted of changing the year to a numeric value and dropping redundant state infomration (state_po) followed by one-hot-encoding the remaining string type objects. After one-hot-encoding, the data was appended to the census_data and the duplicate columns were dropped. 

The model is an SVC with 'rbf' kernell. Years 2012 and 2016 data were used to Test and Train the data. The results are listed below.

Test Acc: 0.817

           precision    recall  f1-score   support

    Democrat       0.70      0.20      0.31       311
  Republican       0.82      0.98      0.89      1191

    accuracy                           0.82      1502
   macro avg       0.76      0.59      0.60      1502
weighted avg       0.80      0.82      0.77      1502


## Predict year 2020 results using the Model:
After training and testing the model, we were able to feed 2020 data through the model and get the predicted winners.



# Logisitcs Regression


# Neural Netork

# Random Forest



# Conclusion


Data sources: 
Census.gov: API call to get demographic data by year (2016 and 2020)
Xxx: winners by state
http://www.structnet.com/instructions/zip_min_max_by_state.html: Zip Code by state



Libraries:
SKlearn
Matplotlib
Numpy
Census
Tensorflow

Additional Files 
