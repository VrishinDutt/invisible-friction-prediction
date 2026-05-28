import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 20000

task_types = [
    "academic", "household", "communication",
    "finance", "health", "personal_admin"
]

time_of_day = ["morning", "afternoon", "evening", "night"]
deadline_pressure = ["low", "medium", "high"]
reward_clarity = ["unclear", "moderate", "clear"]
environment_noise = ["low", "medium", "high"]
device_distraction = ["low", "medium", "high"]
social_obligation = ["individual", "family", "college", "work"]

df = pd.DataFrame({
    "task_type": np.random.choice(task_types, N, p=[0.25, 0.2, 0.2, 0.1, 0.1, 0.15]),
    "estimated_duration": np.random.randint(5, 121, N),
    "deadline_pressure": np.random.choice(deadline_pressure, N, p=[0.45, 0.35, 0.2]),
    "time_of_day": np.random.choice(time_of_day, N, p=[0.25, 0.3, 0.3, 0.15]),
    "energy_level": np.random.randint(1, 11, N),
    "mood_level": np.random.randint(1, 11, N),
    "task_familiarity": np.random.choice(["new", "repeated"], N, p=[0.35, 0.65]),
    "reward_clarity": np.random.choice(reward_clarity, N, p=[0.3, 0.45, 0.25]),
    "consequence_severity": np.random.randint(1, 11, N),
    "previous_delay_count": np.random.randint(0, 8, N),
    "environment_noise": np.random.choice(environment_noise, N, p=[0.35, 0.45, 0.2]),
    "device_distraction": np.random.choice(device_distraction, N, p=[0.3, 0.45, 0.25]),
    "social_obligation": np.random.choice(social_obligation, N, p=[0.45, 0.2, 0.2, 0.15])
})

task_weight = {
    "academic": 14,
    "household": 6,
    "communication": 10,
    "finance": 12,
    "health": 8,
    "personal_admin": 11
}

deadline_weight = {"low": 0, "medium": 8, "high": 16}
reward_weight = {"unclear": 12, "moderate": 5, "clear": -5}
noise_weight = {"low": 0, "medium": 5, "high": 10}
distraction_weight = {"low": 0, "medium": 7, "high": 14}
familiarity_weight = {"new": 9, "repeated": 0}
time_weight = {"morning": -5, "afternoon": 0, "evening": 4, "night": 9}

score = (
    df["task_type"].map(task_weight)
    + df["deadline_pressure"].map(deadline_weight)
    + df["reward_clarity"].map(reward_weight)
    + df["environment_noise"].map(noise_weight)
    + df["device_distraction"].map(distraction_weight)
    + df["task_familiarity"].map(familiarity_weight)
    + df["time_of_day"].map(time_weight)
    + df["estimated_duration"] * 0.12
    + df["previous_delay_count"] * 3
    + df["consequence_severity"] * 1.5
    - df["energy_level"] * 2.2
    - df["mood_level"] * 1.8
    + np.random.normal(0, 8, N)
)

df["friction_score"] = score.round(2)

df["friction_level"] = pd.cut(
    df["friction_score"],
    bins=[-100, 35, 65, 150],
    labels=["low", "medium", "high"]
)

output_path = Path("data/invisible_friction_dataset.csv")
output_path.parent.mkdir(exist_ok=True)
df.to_csv(output_path, index=False)

print("Dataset generated successfully.")
print(f"Rows: {len(df)}")
print(f"Saved to: {output_path}")
print("\nFriction level distribution:")
print(df["friction_level"].value_counts())
print("\nSample rows:")
print(df.head())
