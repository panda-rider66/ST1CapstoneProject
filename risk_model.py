# Created by Nafis Khan u3253332
import pandas as pd
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from dataclasses import dataclass

# Data file import
df = pd.read_csv("heart_attack_prediction_dataset2.csv")

# Attribute to be predicted
predict = "HeartAttackRisk"

# pre-processing
# encode object columns to integers
for col in df:
    if df[col].dtype == 'object':
        df[col] = OrdinalEncoder().fit_transform(df[col].values.reshape(-1, 1))

# Dataset to be Predicted, X is all attributes and y is the features
le = preprocessing.LabelEncoder()
Age = le.fit_transform(list(df["Age"]))
Cholesterol = le.fit_transform(list(df["Cholesterol"]))
Heart_Rate = le.fit_transform(list(df["Heart Rate"]))
Diabetes = le.fit_transform(list(df["Diabetes"]))  #  Whether the patient has diabetes (1: Yes, 0: No)
Family_History = le.fit_transform(list(df["Family History"]))  # Family history of heart-related problems (1: Yes, 0: No)
Smoking = le.fit_transform(list(df["Smoking"]))  # Smoking status of the patient (1: Smoker, 0: Non-smoker)
Obesity = le.fit_transform(list(df["Obesity"]))  # Obesity status of the patient (1: Obese, 0: Not obese)
Exercise_Hours_Per_Week = le.fit_transform(list(df["Exercise Hours Per Week"]))
Previous_Heart_Problems = le.fit_transform(list(df["Previous Heart Problems"]))  # Previous heart problems of the patient (1: Yes, 0: No)
Medication_Use = le.fit_transform(list(df["Medication Use"]))  # Medication usage by the patient (1: Yes, 0: No)
Stress_Level = le.fit_transform(list(df["Stress Level"]))  # Stress level reported by the patient (1-10)
Sedentary_Hours_Per_Day = le.fit_transform(list(df["Sedentary Hours Per Day"]))
Income = le.fit_transform(list(df["Income"]))
BMI = le.fit_transform(list(df["BMI"]))
Triglycerides = le.fit_transform(list(df["Triglycerides"]))
Physical_Activity_Days_Per_Week = le.fit_transform(list(df["Physical Activity Days Per Week"]))
Sleep_Hours_Per_Day = le.fit_transform(list(df["Sleep Hours Per Day"]))
Heart_Attack_Risk = le.fit_transform(list(df["HeartAttackRisk"]))  # Presence of heart attack risk (1: Yes, 0: No)

x = list(zip(Age, Cholesterol, Heart_Rate, Diabetes, Family_History, Smoking, Obesity, Exercise_Hours_Per_Week,
             Previous_Heart_Problems, Medication_Use, Stress_Level, Sedentary_Hours_Per_Day, Income, BMI, Triglycerides,
             Physical_Activity_Days_Per_Week, Sleep_Hours_Per_Day))
y = list(Heart_Attack_Risk)

# Model Test/Train
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = "accuracy"

# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
# 0.2 means 80% training 20% testing, with higher data it already has seen that information and knows
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=seed)


# Prediction class for import
@dataclass(eq=True, frozen=True, order=True)
class Prediction:
    best_model = GaussianNB()
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    model_accuracy = accuracy_score(y_test, y_pred)