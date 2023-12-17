import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

print("Reading CSV file...")
loan_dataset = pd.read_csv("loan.csv", delimiter='\t')
print("CSV file read successfully.")

print("First few rows of the dataset:")
print(loan_dataset.head())

print("Shape of the dataset:", loan_dataset.shape)
print("Summary statistics:")
print(loan_dataset.describe())


print("Number of missing values in the dataset:")
print(loan_dataset.isnull().sum())

loan_dataset = loan_dataset.dropna()

print("Number of missing values after dropping rows:")
print(loan_dataset.isnull().sum())

loan_dataset.replace({"Loan_Status": {"N": 0, "Y": 1}}, inplace=True)
loan_dataset.replace({"Married": {"No": 0, "Yes": 1},
                      "Gender": {"Male": 1, "Female": 0},
                      "Self_Employed": {"No": 0, "Yes": 1},
                      "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
                      "Education": {"Graduate": 1, "Not Graduate": 0},
                      "Dependents": {"0": 0, "1": 1}
                      }, inplace=True)

X = loan_dataset.drop(columns=["Loan_ID", "Loan_Status"], axis=1).values
Y = loan_dataset["Loan_Status"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.1, random_state=2)

classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, Y_train)


print("Total data: ", X_scaled.shape)
print("Train data: ", X_train.shape)
print("Test data: ", X_test.shape)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data: ", training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on test data: ", test_data_accuracy)

joblib.dump(classifier, 'loan_prediction_model.pkl')

# Prediction and approval code for a specific data point
index_to_check = 0

if 0 <= index_to_check < len(Y_test):
    X_new = X_test[index_to_check].reshape(1, -1)
    prediction = classifier.predict(X_new)

    print("Actual Loan Status:", "Loan Approved" if Y_test[index_to_check] == 1 else "Loan Not Approved")
    print("Predicted Loan Status:", "Loan Approved" if prediction[0] == 1 else "Loan Not Approved")
else:
    print("Invalid index for Y_test.")
