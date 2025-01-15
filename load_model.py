import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from utils import StepCounter, preprocess_data


def main():
    step_counter = StepCounter()

    # Load the saved model
    step_counter.show_step("Loading saved model")
    try:
        model = joblib.load("titanic_model.pkl")
    except FileNotFoundError:
        print("Error: Model file 'titanic_model.pkl' not found!")
        return

    # Load the dataset
    step_counter.show_step("Loading dataset")
    try:
        df = pd.read_csv("train.csv")
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

    # Sample predictions
    step_counter.show_step("Sample Predictions")
    sample_indices = np.random.randint(0, len(X), 5)
    sample_data = df.iloc[sample_indices]
    sample_X = X_scaled[sample_indices]
    sample_predictions = model.predict(sample_X)

    for idx, (_, passenger) in enumerate(sample_data.iterrows()):
        print(f"\nPassenger {idx + 1}:")
        print(f"Actual survival: {'Yes' if passenger['Survived'] == 1 else 'No'}")
        print(f"Predicted survival: {'Yes' if sample_predictions[idx] == 1 else 'No'}")


if __name__ == "__main__":
    main()
