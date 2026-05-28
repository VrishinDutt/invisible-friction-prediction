import pandas as pd
import joblib

model = joblib.load(
    "outputs/best_model.joblib"
)

print("\nINVISIBLE FRICTION PREDICTOR\n")

task_type = input(
    "Task Type (academic/household/communication/finance/health/personal_admin): "
)

duration = int(
    input("Estimated Duration (minutes): ")
)

deadline = input(
    "Deadline Pressure (low/medium/high): "
)

time_of_day = input(
    "Time of Day (morning/afternoon/evening/night): "
)

energy = int(
    input("Energy Level (1-10): ")
)

mood = int(
    input("Mood Level (1-10): ")
)

task_familiarity = input(
    "Task Familiarity (new/repeated): "
)

reward = input(
    "Reward Clarity (unclear/moderate/clear): "
)

severity = int(
    input("Consequence Severity (1-10): ")
)

delays = int(
    input("Previous Delay Count: ")
)

noise = input(
    "Environment Noise (low/medium/high): "
)

distraction = input(
    "Device Distraction (low/medium/high): "
)

obligation = input(
    "Social Obligation (individual/family/college/work): "
)

sample = pd.DataFrame([{
    "task_type": task_type,
    "estimated_duration": duration,
    "deadline_pressure": deadline,
    "time_of_day": time_of_day,
    "energy_level": energy,
    "mood_level": mood,
    "task_familiarity": task_familiarity,
    "reward_clarity": reward,
    "consequence_severity": severity,
    "previous_delay_count": delays,
    "environment_noise": noise,
    "device_distraction": distraction,
    "social_obligation": obligation
}])

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

classes = model.named_steps["model"].classes_

print("\nRESULT")
print("-" * 40)

print(
    f"Predicted Friction: {prediction.upper()}"
)

print("\nProbabilities:")

for cls, prob in zip(classes, probabilities):
    print(
        f"{cls:<10}: {prob:.2%}"
    )
print("\nDone.")
prob_dict = dict(zip(classes, probabilities))

friction_index = (
    prob_dict.get("low", 0) * 20 +
    prob_dict.get("medium", 0) * 55 +
    prob_dict.get("high", 0) * 90
)

print(f"\nFriction Index: {friction_index:.1f}/100")
print("\nSuggested Intervention")

if prediction == "high":
    print("• Break the task into a 5-minute starting step")
    print("• Reduce device distractions before starting")
    print("• Make the reward/outcome clearer")
elif prediction == "medium":
    print("• Start with a smaller version of the task")
    print("• Remove one distraction source")
    print("• Schedule it during a higher-energy period")
else:
    print("• Task is likely manageable in current conditions")
    print("• Proceed without overplanning")
print("\nLikely Contributors")

if energy <= 4:
    print("• Low energy")

if mood <= 4:
    print("• Low mood")

if delays >= 3:
    print("• Repeated postponement")

if deadline == "high":
    print("• High deadline pressure")

if distraction == "high":
    print("• High device distraction")

if reward == "unclear":
    print("• Unclear reward")

print("\nDone.")
