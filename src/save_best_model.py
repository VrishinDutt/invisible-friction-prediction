import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sampling import stratified_sample

df = pd.read_csv("data/invisible_friction_dataset.csv")

train_full, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=42,
    stratify=df["friction_level"]
)

train_df = stratified_sample(
    train_full,
    sample_size=5000
)

X_train = train_df.drop(
    columns=["friction_level", "friction_score"]
)

y_train = train_df["friction_level"]

categorical_features = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(
    pipeline,
    "outputs/best_model.joblib"
)

print("Model saved.")
