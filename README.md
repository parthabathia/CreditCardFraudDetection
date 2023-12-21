# Credit Card Fraud Detection

This Python code is an example of a machine learning workflow for credit card fraud detection. Here's a breakdown of the code:

1. **Import Libraries:**
   - `numpy`, `pandas`, `matplotlib.pyplot`, and `seaborn` are imported for data manipulation and visualization.
   - `train_test_split` is imported from `sklearn.model_selection` for splitting the dataset.
   - `LinearRegression` and `LogisticRegression` are imported from `sklearn.linear_model` for building the machine learning models.
   - `metrics` is imported from `sklearn` to evaluate the model performance.

2. **Load and Explore Data:**
   - The code reads a CSV file (`creditcard.csv`) containing credit card transaction data using `pd.read_csv`.
   - The last few rows of the dataset are displayed using `creditcard_dataset.tail()`.

3. **Data Analysis:**
   - The code calculates the correlation matrix for the dataset using `creditcard_dataset.corr()` and visualizes it using a heatmap with Seaborn.

4. **Data Preprocessing:**
   - Descriptive statistics and the count of null values for each column are displayed using `creditcard_dataset.describe()` and `creditcard_dataset.isnull().sum()`.

5. **Data Sampling:**
   - The code separates legitimate transactions (`legit`) and fraudulent transactions (`fraud`) from the dataset.
   - Randomly samples 500 instances from the legitimate transactions (`sample_dataset`).
   - Combines the sampled dataset with the fraudulent transactions (`creditcard_sample`) using `pd.concat`.

6. **Data Splitting:**
   - Splits the dataset into training and testing sets using `train_test_split` with a test size of 20%.

7. **Model Training:**
   - Initializes a logistic regression model (`model_log`) and fits it to the training data using `model_log.fit()`.

8. **Model Evaluation on Training Data:**
   - Generates predictions on the training data and calculates the accuracy using `metrics.accuracy_score`.

9. **Model Evaluation on Testing Data:**
   - Generates predictions on the testing data and calculates the accuracy using `metrics.accuracy_score`.

10. **Prediction for New Input:**
    - Defines a new input (`new_input`) representing a credit card transaction.
    - Uses the trained logistic regression model to predict whether the transaction is fraudulent or not (`prediction_new_input`).

The main focus of this code is on building a logistic regression model for credit card fraud detection and evaluating its performance on a sampled dataset. The new input at the end is used to demonstrate how the model can make predictions for unseen data.
