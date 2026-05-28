import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sampling import (
    simple_random_sample,
    stratified_sample,
    balanced_stratified_sample,
    cluster_sample
)

df = pd.read_csv("data/invisible_friction_dataset.csv")

samples = {
    "Population": df,
    "Simple Random": simple_random_sample(df, 5000),
    "Proportional Stratified": stratified_sample(df, 5000),
    "Balanced Stratified": balanced_stratified_sample(df, 5000),
    "Cluster": cluster_sample(df)
}

out = Path("outputs/statistical_analysis")
out.mkdir(parents=True, exist_ok=True)

for name, sample in samples.items():

    counts = sample["friction_level"].value_counts()

    plt.figure(figsize=(5,5))

    plt.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%"
    )

    plt.title(name)

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_distribution.png"
    )

    plt.savefig(
        out / filename,
        dpi=300
    )

    plt.close()

print("Sampling visualizations generated.")
