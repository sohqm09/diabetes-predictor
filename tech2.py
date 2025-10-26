import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.utils.class_weight import compute_class_weight

# Load dataset
data = pd.read_csv("diabetes (2).csv", na_values=["?"])  # Handle missing values directly

# Convert to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Select relevant columns
selected_features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Handle missing values by imputing median values
data.fillna(data.median(numeric_only=True), inplace=True)

# Feature Engineering
data["BMI_Age"] = data["BMI"] * data["Age"]
data["Glucose_Insulin"] = data["Glucose"] * data["Insulin"]
data["BP_BMI"] = data["BloodPressure"] * data["BMI"]

# Update selected features to include new ones
selected_features += ["BMI_Age", "Glucose_Insulin", "BP_BMI"]

# Define X and y
X = data[selected_features]
y = data["Outcome"]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Compute class weights
class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
scale_pos_weight_value = class_weights[1] / class_weights[0]  # Corrected formula

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight_value, random_state=42)

# Perform hyperparameter tuning
grid_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=20, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Use the best model
model = grid_search.best_estimator_

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

# Function to predict diabetes and estimate time to diabetes onset
def predict_diabetes():
    print("Enter the following details:")
    user_input = []
    
    for feature in selected_features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                user_input.append(value)
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    # Load model and scaler
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Scale input and predict
    user_input_scaled = scaler.transform([user_input])
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]  # Get probability of diabetes

    if prediction == 1:
        print("The person is likely to have diabetes.")
    else:
        print("The person is unlikely to have diabetes.")

        # Estimate years to diabetes onset
        age = user_input[selected_features.index("Age")]
        estimated_years = (1 - probability) * (80 - age)  # Assuming max age 80
        print(f"Based on current health data, the person may develop diabetes in approximately {estimated_years:.1f} years.")

    # Collect user feedback
    feedback = input("Was this prediction accurate? (yes/no): ").strip().lower()
    with open("user_feedback.log", "a") as log_file:
        log_file.write(f"Input: {user_input}, Prediction: {prediction}, Feedback: {feedback}\n")
    
    print("Thank you for your feedback!")

# Uncomment the line below to enable user input mode
# predict_diabetes()