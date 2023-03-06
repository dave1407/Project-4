# Logistic Regression Model
## Import data from DB for modeling :
* Import the database from MongoDB for this modeling
* Read the database and keep the columns necessary for modeling and drop the unnecessary columns from the database.

## Build Logistic Regression Model :
* To do Logistic REgression Modeling , We need target and features column 
    * Make the target column as a binary indicator
    * Select the columns that is going to have an effect in predicting the winner of the county elections as the features.
    * Pick data for years 2012 and 2016, for training a model.
    * Split them into training and testing data.
    * Create a Logistic Regression Model.
    * Fit(train) our model by using the training data.
    * Show the model accuracy for training and test data.

## Predict year 2020 results using the Model:
* Show the Actual and Predicted values of the winner and their difference in the dataframe.
* use a scatter Plot to show the prediction accuracy.
    * When red(Actual winner) and green(predicted winner) are together in the below plot, that means prediction matches Actual results.
    