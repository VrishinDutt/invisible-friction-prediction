import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/invisible_friction_dataset.csv")

numeric = df.select_dtypes(include=["int64", "float64"])

corr = numeric.corr()

out = Path("outputs/statistical_analysis")
out.mkdir(parents=True, exist_ok=True)

corr.to_csv(out / "correlation_matrix.csv")

plt.figure(figsize=(10,8))
plt.imshow(corr)
plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.tight_layout()

plt.savefig(
    out / "correlation_matrix.png",
    dpi=300
)

print(corr)
print("\nSaved correlation matrix.")

