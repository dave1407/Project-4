# Logistic Regression Model
# pip install imblearn
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