# student-performance-prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("student_data.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Features and Target
X = df[['Study_Hours',
        'Attendance',
        'Previous_Marks',
        'Assignment_Score']]

y = df['Final_Marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

# Save model
joblib.dump(model, "student_model.pkl")

# Load model
loaded_model = joblib.load("student_model.pkl")

# User input
print("\nEnter the details:")
study_hours = float(input("Study Hours: "))
attendance = float(input("Attendance Percentage: "))
previous_marks = float(input("Previous Marks: "))
assignment_score = float(input("Assignment Score: "))

# Predict
result = loaded_model.predict(
    [[study_hours,
      attendance,
      previous_marks,
      assignment_score]]
)

print("\nPredicted Final Marks:", round(result[0], 2))
