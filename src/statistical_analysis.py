import pandas as pd
from pathlib import Path

df = pd.read_csv("data/invisible_friction_dataset.csv")

out = Path("outputs/statistical_analysis")
out.mkdir(parents=True, exist_ok=True)

summary = df.describe(include="all")

summary.to_csv(
    out / "descriptive_statistics.csv"
)

print(summary)
print("\nSaved descriptive statistics.")

