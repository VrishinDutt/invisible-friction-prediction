import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("outputs/model_results.csv")

out = Path("outputs/plots")
out.mkdir(parents=True, exist_ok=True)

df["label"] = df["sampling_method"] + "\n" + df["model"]

plt.figure(figsize=(12, 6))
plt.bar(df["label"], df["f1_score"])
plt.xticks(rotation=75, ha="right")
plt.ylabel("F1-score")
plt.title("F1-score Across Sampling Methods and Models")
plt.tight_layout()
plt.savefig(out / "f1_score_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
plt.bar(df["label"], df["model_error"])
plt.xticks(rotation=75, ha="right")
plt.ylabel("Model Error")
plt.title("Model Error Across Sampling Methods and Models")
plt.tight_layout()
plt.savefig(out / "model_error_comparison.png", dpi=300)
plt.close()

best = df.sort_values("f1_score", ascending=False).groupby("sampling_method").head(1)

plt.figure(figsize=(8, 5))
plt.bar(best["sampling_method"], best["f1_score"])
plt.ylabel("Best F1-score")
plt.title("Best Model per Sampling Method")
plt.tight_layout()
plt.savefig(out / "best_sampling_method.png", dpi=300)
plt.close()

print("Plots saved to outputs/plots/")
