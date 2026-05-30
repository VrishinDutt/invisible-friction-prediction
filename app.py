import pandas as pd
import joblib
import streamlit as st

model = joblib.load("outputs/best_model.joblib")

st.set_page_config(
    page_title="Invisible Friction Predictor",
    page_icon="🧠",
    layout="centered"
)

st.title("Invisible Friction Predictor")
st.write("Predict how mentally difficult a task may feel before starting it.")

task_type = st.selectbox(
    "Task Type",
    ["academic", "household", "communication", "finance", "health", "personal_admin"]
)

duration = st.slider("Estimated Duration (minutes)", 5, 120, 30)

deadline = st.selectbox("Deadline Pressure", ["low", "medium", "high"])

time_of_day = st.selectbox(
    "Time of Day",
    ["morning", "afternoon", "evening", "night"]
)

energy = st.slider("Energy Level", 1, 10, 5)

mood = st.slider("Mood Level", 1, 10, 5)

task_familiarity = st.selectbox("Task Familiarity", ["new", "repeated"])

reward = st.selectbox("Reward Clarity", ["unclear", "moderate", "clear"])

severity = st.slider("Consequence Severity", 1, 10, 5)

delays = st.slider("Previous Delay Count", 0, 7, 2)

noise = st.selectbox("Environment Noise", ["low", "medium", "high"])

distraction = st.selectbox("Device Distraction", ["low", "medium", "high"])

obligation = st.selectbox(
    "Social Obligation",
    ["individual", "family", "college", "work"]
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

if st.button("Predict Friction"):
    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    classes = model.named_steps["model"].classes_

    prob_dict = dict(zip(classes, probabilities))

    friction_index = (
        prob_dict.get("low", 0) * 20 +
        prob_dict.get("medium", 0) * 55 +
        prob_dict.get("high", 0) * 90
    )

    st.subheader(f"Predicted Friction: {prediction.upper()}")
    st.metric("Friction Index", f"{friction_index:.1f}/100")

    st.write("### Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Friction Level": classes,
        "Probability": probabilities
    })

    st.dataframe(prob_df, width="stretch")

    st.write("### Likely Contributors")

    contributors = []

    if energy <= 4:
        contributors.append("Low energy")
    if mood <= 4:
        contributors.append("Low mood")
    if delays >= 3:
        contributors.append("Repeated postponement")
    if deadline == "high":
        contributors.append("High deadline pressure")
    if distraction == "high":
        contributors.append("High device distraction")
    if reward == "unclear":
        contributors.append("Unclear reward")
    if noise == "high":
        contributors.append("High environment noise")
    if time_of_day == "night":
        contributors.append("Night-time task timing")

    if contributors:
        for item in contributors:
            st.write(f"- {item}")
    else:
        st.write("- No major friction contributors detected.")

    st.write("### Suggested Intervention")

    if prediction == "high":
        st.warning(
            "Break the task into a 5-minute starting step, reduce distractions, and clarify the immediate reward."
        )
    elif prediction == "medium":
        st.info(
            "Start with a smaller version of the task and schedule it during a higher-energy period."
        )
    else:
        st.success(
            "The task is likely manageable under current conditions."
        )
