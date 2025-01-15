import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Add this import


class StepCounter:
    def __init__(self):
        self.step = 1

    def show_step(self, title: str):
        print(f"\n {self.step}. {title}")
        self.step += 1


step_counter = StepCounter()


# Load the Iris dataset
iris = load_iris()
step_counter.show_step("Iris dataset loaded")

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

step_counter.show_step("Data splitted")
shape_data = pd.DataFrame(
    {
        "Set": ["X_train", "X_test", "y_train", "y_test"],
        "Shape": [X_train.shape, X_test.shape, y_train.shape, y_test.shape],
    }
)
print(shape_data)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
step_counter.show_step("Random Forest initialized")

rf_model.fit(X_train_scaled, y_train)
step_counter.show_step("Model trained")

# Save the trained model
joblib.dump(rf_model, "rf_model.pkl")
step_counter.show_step("Model saved to rf_model.pkl")

# Make predictions
y_pred = rf_model.predict(X_test_scaled)
step_counter.show_step("Predictions made")

# Print model performance metrics
step_counter.show_step("Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Create confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
step_counter.show_step("Confusion Matrix:")
print(cm)
