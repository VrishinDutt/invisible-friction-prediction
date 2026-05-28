import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sampling import (
    simple_random_sample,
    stratified_sample,
    balanced_stratified_sample,
    cluster_sample,
    show_sample_distribution
)

DATA_PATH = Path("data/invisible_friction_dataset.csv")
OUTPUT_PATH = Path("outputs/model_results.csv")
OUTPUT_PATH.parent.mkdir(exist_ok=True)


def evaluate_model(model_name, sampling_name, model, train_df, test_df):
    target = "friction_level"

    X_train = train_df.drop(columns=["friction_level", "friction_score"])
    y_train = train_df[target]

    X_test = test_df.drop(columns=["friction_level", "friction_score"])
    y_test = test_df[target]
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_features = X_train.select_dtypes(exclude=["object", "string"]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numerical_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
    recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    error = 1 - accuracy

    cm = confusion_matrix(y_test, predictions, labels=["low", "medium", "high"])

    cm_path = Path(f"outputs/confusion_matrices/{sampling_name}_{model_name}.csv")
    cm_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        cm,
        index=["actual_low", "actual_medium", "actual_high"],
        columns=["pred_low", "pred_medium", "pred_high"]
    ).to_csv(cm_path)

    return {
        "sampling_method": sampling_name,
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "model_error": round(error, 4),
        "confusion_matrix_file": str(cm_path)
    }


def main():
    df = pd.read_csv(DATA_PATH)

    train_full, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["friction_level"]
    )

    samples = {
        "simple_random": simple_random_sample(train_full, sample_size=5000),
        "proportional_stratified": stratified_sample(train_full, sample_size=5000),
        "balanced_stratified": balanced_stratified_sample(train_full, sample_size=5000),
        "cluster": cluster_sample(train_full, cluster_column="task_type", n_clusters=3)
    }
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=6),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=12,
            class_weight="balanced"
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42)
    }

    results = []

    for sampling_name, sample_df in samples.items():
        show_sample_distribution(sample_df, sampling_name)

        for model_name, model in models.items():
            print(f"Training {model_name} using {sampling_name} sampling...")
            result = evaluate_model(
                model_name,
                sampling_name,
                model,
                sample_df,
                test_df
            )
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal Results:")
    print(results_df)
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
