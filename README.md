This project involves analyzing the quality of wine using machine learning models. The analysis includes data preprocessing, 
model building, and evaluation for both classification and regression approaches. Below are the key steps carried out in the project:

1. Data Quality Assessment and Exploratory Data Analysis (EDA):
Evaluate the quality of the data.
Perform exploratory data analysis to understand patterns and relationships.
Prepare the data for analysis, including cleaning, feature scaling, and handling missing values.
Split the data into training and testing sets.

2. Building Classification Models:
Develop two different classification models to predict wine quality based on the training set.
Evaluate the performance of the models on the test set using appropriate metrics (e.g., accuracy, precision, recall, F1-score).

3. Building Regression Models:
Treat the "quality" variable as a continuous variable.
Build two regression models to estimate wine quality on the training set.
Evaluate the models on the test set using metrics such as Mean Absolute Error (MAE) or Mean Squared Error (MSE).
Round the predicted values to the nearest whole number for comparison with the true quality ratings (e.g., a predicted quality of 5.34 is rounded to 5, while 5.69 is rounded to 6).

4. Model Comparison:
Compare the performance of classification models with regression models in predicting wine quality.
Discuss strengths and weaknesses of each approach.

5. Clustering Analysis:
Using features 1-11, group the data into clusters using two different clustering algorithms (e.g., K-Means and Hierarchical Clustering) based on the training set.
Analyze the resulting clusters to identify any meaningful patterns or insights.
