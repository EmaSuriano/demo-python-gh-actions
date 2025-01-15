import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from utils import StepCounter, preprocess_data
import os
from sklearn.metrics import classification_report


def main():
    step_counter = StepCounter()
    current_dir = os.path.dirname(__file__)

    # Load the saved model
    step_counter.show_step("Loading saved model")
    try:
        model = joblib.load(os.path.join(current_dir, "titanic_model.pkl"))
    except FileNotFoundError:
        print("Error: Model file 'titanic_model.pkl' not found!")
        return

    # Load the dataset
    step_counter.show_step("Loading dataset")
    try:
        df = pd.read_csv(os.path.join(current_dir, "train.csv"))
        print(f"Dataset shape: {df.shape}")
    except FileNotFoundError:
        print("Error: Dataset file 'train.csv' not found!")
        return

    # Extract target variable
    y = df["Survived"]

    # Preprocess features
    step_counter.show_step("Preprocessing data")
    X = preprocess_data(df)
    print("Features after preprocessing:", X.columns.tolist())

    # Encode categorical variables
    step_counter.show_step("Encoding categorical variables")
    categorical_features = ["Sex", "Embarked", "Title"]
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])

    # Scale features
    step_counter.show_step("Scaling features")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Evaluate the model
    step_counter.show_step("Evaluating model")
    y_pred = model.predict(X_scaled)
    report = classification_report(y, y_pred, target_names=["Not Survived", "Survived"])
    print("\nClassification Report:\n", report)


if __name__ == "__main__":
    main()
