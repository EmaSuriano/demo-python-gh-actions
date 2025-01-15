import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from utils import StepCounter, preprocess_data
import joblib
from time import time
import os


def main():
    start_time = time()
    step_counter = StepCounter()

    # Load the dataset
    step_counter.show_step("Loading Titanic dataset")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_dir, "train.csv"))
    print(f"Dataset shape: {df.shape}")

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

    # Split the data
    step_counter.show_step("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    shape_data = pd.DataFrame(
        {
            "Set": ["X_train", "X_test", "y_train", "y_test"],
            "Shape": [X_train.shape, X_test.shape, y_train.shape, y_test.shape],
        }
    )
    print(shape_data)

    # Scale features
    step_counter.show_step("Scaling features")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train model
    step_counter.show_step("Initializing Random Forest Classifier")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,  # Use all available cores
    )

    step_counter.show_step("Training model")
    rf_model.fit(X_train_scaled, y_train)

    # Save model
    step_counter.show_step("Saving model")
    joblib.dump(rf_model, os.path.join(current_dir, "titanic_model.pkl"))

    # Make predictions
    step_counter.show_step("Making predictions")
    y_pred = rf_model.predict(X_test_scaled)

    # Evaluate model
    step_counter.show_step("Model Performance Metrics")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    step_counter.show_step("Feature Importance Analysis")
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # Print total execution time
    execution_time = time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
