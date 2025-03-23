# Titanic Survival Prediction

## Overview
This project builds a machine learning model to predict whether a passenger survived the Titanic disaster based on their attributes using a **Jupyter Notebook**.

## Task Objectives
- Develop a classification model to predict survival.
- Preprocess the dataset by handling missing values, encoding categorical variables, and normalizing numerical data.
- Train a **Random Forest Classifier** and evaluate its performance.
- Visualize important features influencing survival.

## Dataset
The dataset contains the following key features:
- `Pclass` (Ticket class: 1st, 2nd, 3rd)
- `Sex` (Gender of the passenger)
- `Age` (Age of the passenger, with some missing values handled)
- `SibSp` (Number of siblings/spouses aboard)
- `Parch` (Number of parents/children aboard)
- `Fare` (Ticket fare)
- `Embarked` (Port of embarkation)

## Steps in the Notebook

### Step 1: Import Required Libraries
- Import `pandas`, `numpy`, `seaborn`, `matplotlib`, and `sklearn` for data processing and visualization.

### Step 2: Load Dataset
- Read the dataset and display basic information.

### Step 3: Data Preprocessing
1. Handle missing values:
   - Drop `Cabin` due to excessive missing data.
   - Fill missing `Age` and `Fare` values with the median.
2. Encode categorical variables (`Sex`, `Embarked`).
3. Normalize numerical features (`Age`, `Fare`).

### Step 4: Data Splitting
- Split the dataset into training and testing sets (80/20 split).

### Step 5: Train Machine Learning Model
- Train a **Random Forest Classifier** with 100 estimators.

### Step 6: Model Evaluation
- Evaluate the model using **accuracy, precision, recall, and F1-score**.

### Step 7: Data Visualization
- Plot feature importance to understand the most influential factors in survival prediction.

## Steps to Run the Project
1. Clone the repository:
   ```sh
   git clone <repository_link>
   ```
2. Navigate to the project directory:
   ```sh
   cd Titanic-Survival-Prediction
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Jupyter Notebook:
   ```sh
   jupyter notebook titanic_model.ipynb
   ```
5. Execute all steps in the notebook to train and evaluate the model.

## Results
- The model achieved strong predictive performance.
- Feature importance analysis provides insights into survival factors.

## Future Enhancements
- Try different models like Logistic Regression, XGBoost, or Neural Networks.
- Apply hyperparameter tuning for better accuracy.
- Perform cross-validation to improve model robustness.

## Author
Developed as part of the GrowthLink Internship Assignment.

