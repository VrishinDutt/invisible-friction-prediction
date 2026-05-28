import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

train_df = stratified_sample(train_full, sample_size=5000)

X_train = train_df.drop(columns=["friction_level", "friction_score"])
y_train = train_df["friction_level"]

categorical_features = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
numerical_features = X_train.select_dtypes(exclude=["object", "string"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

model = LogisticRegression(max_iter=1000)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
classes = pipeline.named_steps["model"].classes_
coefficients = pipeline.named_steps["model"].coef_

out = Path("outputs/explainability")
out.mkdir(parents=True, exist_ok=True)

for i, cls in enumerate(classes):
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients[i]
    })

    coef_df["absolute_value"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("absolute_value", ascending=False).head(15)

    coef_df.to_csv(out / f"logistic_coefficients_{cls}.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(coef_df["feature"], coef_df["coefficient"])
    plt.xlabel("Coefficient Value")
    plt.title(f"Top Predictors for {cls.upper()} Friction")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out / f"logistic_coefficients_{cls}.png", dpi=300)
    plt.close()

print("Explainability outputs saved to outputs/explainability/")
