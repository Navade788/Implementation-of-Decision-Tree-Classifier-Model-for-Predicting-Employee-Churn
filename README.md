# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

```
Name: S.Navadeep
Reg.No: 212224230180
```

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the employee churn dataset.
2. Split the data into training and testing sets.
3. Train the Decision Tree Classifier model.
4. Predict churn and evaluate model accuracy.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: RANJANI K
RegisterNumber:  212224230220
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

data = pd.DataFrame({
    'Age': [25,30,28,35,40,45,50,26,38,29,33,37,48,42,31,27,36,41,39,34],
    'Salary': [40000,50000,45000,60000,80000,70000,90000,42000,65000,48000,
               52000,61000,72000,81000,53000,47000,58000,79000,68000,55000],
    'Experience': [1,3,2,5,8,6,10,1,7,2,4,5,8,9,3,2,6,8,7,4],
    'Department': ['Sales','HR','IT','Sales','Finance','IT','HR','Finance','Sales','IT',
                   'Finance','IT','Sales','HR','Finance','Sales','IT','HR','Finance','IT'],
    'Churn': ['No','No','Yes','No','No','Yes','No','Yes','No','Yes',
              'Yes','No','No','No','Yes','Yes','No','No','Yes','No']
})

data.to_csv("employee_churn.csv", index=False)

data = pd.read_csv("employee_churn.csv")
data = pd.get_dummies(data, drop_first=True)

X = data.drop("Churn_Yes", axis=1)
y = data["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:
<img width="616" height="420" alt="Screenshot 2025-10-08 155141" src="https://github.com/user-attachments/assets/3bf11e58-f120-4d2a-bdaa-e1816cedb584" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
