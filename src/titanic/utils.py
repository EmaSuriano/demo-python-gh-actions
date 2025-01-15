import pandas as pd


class StepCounter:
    def __init__(self):
        self.step = 1

    def show_step(self, title: str):
        print(f"\n{self.step}. {title}")
        self.step += 1


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Titanic dataset consistently with training preprocessing
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()

    # Extract titles from names
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    # Group rare titles
    rare_titles = [
        "Lady",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ]
    df["Title"] = df["Title"].replace(rare_titles, "Rare")
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")

    # Fill missing ages with median age for each title
    df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

    # Create family size feature
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Create is_alone feature
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Fill missing embarked with most common value
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Fill missing Fare with median
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Select features used in training
    features = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title",
    ]

    return df[features]
