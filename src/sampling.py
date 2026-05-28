import pandas as pd


def simple_random_sample(df, sample_size=5000):
    return df.sample(n=sample_size, random_state=42).reset_index(drop=True)


def stratified_sample(df, sample_size=5000):
    # Stratify by both friction level and task type.
    # This preserves rare high-friction/task-category combinations.
    fractions = sample_size / len(df)

    sample = (
        df.groupby(["friction_level", "task_type"], group_keys=False)
        .sample(frac=fractions, random_state=42)
        .reset_index(drop=True)
    )

    return sample


def balanced_stratified_sample(df, sample_size=5000):
    classes = df["friction_level"].unique()
    per_class = sample_size // len(classes)

    samples = []

    for cls in classes:
        group = df[df["friction_level"] == cls]

        replace_needed = len(group) < per_class

        samples.append(
            group.sample(
                n=per_class,
                replace=replace_needed,
                random_state=42
            )
        )

    return pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

def cluster_sample(df, cluster_column="task_type", n_clusters=3):
    selected_clusters = (
        pd.Series(df[cluster_column].unique())
        .sample(n=n_clusters, random_state=42)
        .tolist()
    )

    return df[df[cluster_column].isin(selected_clusters)].reset_index(drop=True)


def show_sample_distribution(sample, name):
    print(f"\n{name} Sample")
    print("-" * 40)
    print("Rows:", len(sample))

    print("\nColumns:")
    print(sample.columns.tolist())

    print("\nFriction distribution:")
    print(sample["friction_level"].value_counts(normalize=True).round(3))

    print("\nTask type distribution:")
    print(sample["task_type"].value_counts(normalize=True).round(3))
