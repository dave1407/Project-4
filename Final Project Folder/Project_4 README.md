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

# Intro: 
The relationship between demographic factors and election results is complex and multifaceted. While population size can be an important factor in determining election outcomes, it is not the only factor that matters. Other demographic factors such as income, age, and poverty rate can also play a role in shaping voter preferences. 

For example, areas with higher household income may tend to vote for candidates who advocate for policies that benefit the wealthy, while areas with lower household income may be more likely to support candidates who prioritize policies that address economic inequality. Similarly, areas with a higher average age may be more likely to vote for candidates who focus on issues such as retirement benefits and healthcare for seniors, while areas with a younger population may be more concerned with issues such as education and job opportunities.

Overall, understanding the relationship between demographic factors and election results can be an important tool for election campaign managers in developing strategies to target specific voter groups and appeal to their priorities and concerns.

# Modeling approach:
Because we are trying to classify the election reuslts (Republican or Democrate) this project focuses exclusively on supervised machine learning techniques 

# Instructions:

Run  election_ETL.ipynb file firts<br />

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

The group beleived that there may be inbalance in the target data so the SMOTE (Synthetic Minority Oversampling Technique) was applied to the SVM and Logistics Regression models. Running the SVM using SMOTE created an even lower score across all metrics (accuracy, precison and resilience). As a result, SMOTE was removed from the SVM model.  


# Logisitcs Regression
## pip install imblearn
## Import data from DB for modeling :
* Import the database from MongoDB for this modeling
* Read the database and keep the columns necessary for modeling and drop the unnecessary columns from the database.

## Build Logistic Regression Model :
* To do Logistic REgression Modeling , We need target and features column 
    * Make the target column as a binary indicator
    * Select the columns that is going to have an effect in predicting the winner of the county elections as the features.
    * Split them into training and testing data.
    * Create a Logistic Regression Model.
    * Fit(train) our model by using the training data.
    * Show the model accuracy for training and test data.

## Predict results using the Model:
* Show the Predicted values of the winner .
* Use a bar chart to show 'Republican' and 'Democrat' prediction accuracy.
* Use the confusion Matrix to heatmap to show the results.
* Since the imbalance between the data of the 'Republican vs Democrats' is high, we have to do oversampling methods and undersampling methods to find which method gets a better modeling. 

### SMOTE Algorithm
* SMOTE Algorithm has oversampled the minority instances and made it equal to majority class('Republican'). 
* Both categories have equal amount of records. More specifically, the minority class('Democrat') has been increased to the total number of majority class.
* Now see the accuracy and recall results after applying SMOTE algorithm (Oversampling).

### NEVERMISS Algorithm
* The NearMiss Algorithm has undersampled the majority instances and made it equal to majority class('Republican'). 
* Here, the majority class('Republican') has been reduced to the total number of minority class('Democrat'), so that both classes will have equal number of records.

# Neural Network
Neural networks can help computers make intelligent decisions with limited human assistance. This is because they can learn and model the relationships between input and output data that are nonlinear and complex. 

Neural networks consist of multiple layers of interconnected nodes, each building upon the previous layer to refine and optimize the prediction or categorization. This progression of computations through the network is called forward propagation. The input and output layers of a deep neural network are called visible layers. The input layer is where the deep learning model ingests the data for processing, and the output layer is where the final prediction or classification is made.

Another process called backpropagation uses algorithms to calculate errors in predictions and then adjusts the weights and biases of the function by moving backwards through the layers in an effort to train the model. Together, forward propagation and backpropagation allow a neural network to make predictions and correct for any errors accordingly. Over time, the algorithm becomes gradually more accurate.

For this project we used tensorflow library to model a neural network for a US elections dataset based on demographics by US county.

We trained the model with 2012 and 2016 data, getting these results:

![image](https://user-images.githubusercontent.com/10065386/223406694-561c74c9-d000-4cd4-b4c2-bbb313383a20.png)

Then we validated the model with 2020 data, getting these results:

![image](https://user-images.githubusercontent.com/10065386/223406948-d593b7a9-2709-444b-95f9-09072b5ee53d.png)

Eventhough this data does not reflect the actual winner for 2020 US election, who was a Democrat. We can say that the model does its work, as it ended up having an 0.87 average accuracy for the data we feeded.


# Random Forest

Random Forest is an ensemble learning algorithm that constructs multiple decision trees at training time and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. We found this model interesting because we can see how the each feature of the model impacts the predicted results.

The clean data for 2012 and 2016 was retrieved from the mongo database. The columns that were not considered features or targets were dropped. Then training and testing sets were created and the target result was dropped from the data. The model was scaled with a standard scaler and the model was trained.
The results were analyzed and found acceptable with an accuracy of 0.87

The model was used on 2020 data and found that it performed even better than during testing, with an accuracy score of 0.89.


# Conclusion

After careful evaluation of all the models created, we recommend the Random Forest model as the most optimal model.  

# Data sources: 

Census.gov: API call to get demographic data by year (2016 and 2020)
Election winners by county:
http://www.structnet.com/instructions/zip_min_max_by_state.html](https://dataverse.harvard.edu/file.xhtml?fileId=6689930&version=11.0



# A few libraries used:

*SKlearn
*Matplotlib
*Numpy
*Census
*Tensorflow

