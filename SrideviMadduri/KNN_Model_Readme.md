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
