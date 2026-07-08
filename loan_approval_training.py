from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
MODEL_PATH = Path("best_loan_model.pkl")

FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
    "TotalIncome",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUMERIC_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "TotalIncome",
]


def load_dataset() -> pd.DataFrame:
    """Load the training dataset."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found at {TRAIN_PATH}. "
            "Download train.csv from Kaggle or Analytics Vidhya and place it there."
        )

    return pd.read_csv(TRAIN_PATH)


def inspect_dataset(df: pd.DataFrame) -> None:
    """Display basic dataset information."""
    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nTarget distribution:")
    print(df["Loan_Status"].value_counts())


def add_income_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare income-related features used by all models."""
    df = df.copy()
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing for missing values, encoding, and numeric scaling."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ],
        sparse_threshold=0,
    )


def get_models() -> dict:
    """Return the classification models required for comparison."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Naive Bayes": GaussianNB(),
    }


def evaluate_models(X_train, X_valid, y_train, y_valid):
    """Train, evaluate, and compare all models."""
    results = []
    trained_models = {}

    for model_name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_valid)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_valid, y_pred),
                "Precision": precision_score(y_valid, y_pred),
                "Recall": recall_score(y_valid, y_pred),
                "F1-Score": f1_score(y_valid, y_pred),
                "Confusion Matrix": confusion_matrix(y_valid, y_pred).tolist(),
            }
        )

        trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(
        by="F1-Score", ascending=False
    )
    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    return results_df, best_model_name, best_model


def explain_errors(best_model, X_valid, y_valid) -> None:
    """Show false approvals and false rejections for the best model."""
    y_pred = best_model.predict(X_valid)
    analysis_df = X_valid.copy()
    analysis_df["Actual"] = y_valid.values
    analysis_df["Predicted"] = y_pred

    false_approvals = analysis_df[
        (analysis_df["Actual"] == 0) & (analysis_df["Predicted"] == 1)
    ]
    false_rejections = analysis_df[
        (analysis_df["Actual"] == 1) & (analysis_df["Predicted"] == 0)
    ]

    print("\nError Analysis")
    print("\nFalse Approvals (False Positives):")
    print(false_approvals.head())
    print(f"Total false approvals: {len(false_approvals)}")

    print("\nFalse Rejections (False Negatives):")
    print(false_rejections.head())
    print(f"Total false rejections: {len(false_rejections)}")

    print(
        "\nWhy these errors matter: False approvals are risky because they can "
        "approve applicants who may be likely to default, causing financial loss. "
        "False rejections are also important because they deny loans to eligible "
        "applicants, which can reduce customer trust and business opportunities."
    )


def main() -> None:
    df = load_dataset()
    inspect_dataset(df)

    df = df.drop(columns=["Loan_ID"], errors="ignore")
    df = add_income_features(df)
    X = df[FEATURE_COLUMNS]
    y = df["Loan_Status"].map({"N": 0, "Y": 1})

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results_df, best_model_name, best_model = evaluate_models(
        X_train, X_valid, y_train, y_valid
    )

    print("\nModel Comparison Table:")
    print(results_df.to_string(index=False))
    print(f"\nBest-performing model: {best_model_name}")

    explain_errors(best_model, X_valid, y_valid)

    joblib.dump(best_model, MODEL_PATH)
    print(f"\nSaved best model to {MODEL_PATH}")

    if TEST_PATH.exists():
        test_df = pd.read_csv(TEST_PATH)
        test_ids = test_df["Loan_ID"] if "Loan_ID" in test_df.columns else None
        test_df = add_income_features(test_df)
        test_predictions = best_model.predict(test_df[FEATURE_COLUMNS])
        test_output = pd.DataFrame(
            {"Loan_Status": ["Y" if pred == 1 else "N" for pred in test_predictions]}
        )
        if test_ids is not None:
            test_output.insert(0, "Loan_ID", test_ids)
        test_output.to_csv("test_predictions.csv", index=False)
        print("Saved test predictions to test_predictions.csv")


if __name__ == "__main__":
    main()
