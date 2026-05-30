markdown # Invisible Friction Analytics  ## An Inferential Study of Sampling Techniques in Behavioral Machine Learning  Invisible Friction Analytics is an Inferential Statistics and Machine Learning project that predicts the hidden mental resistance people experience before starting everyday tasks.  The project investigates a subtle everyday phenomenon:  > Why do simple tasks sometimes feel disproportionately difficult to begin?  Examples include replying to a message, paying a bill, starting an assignment, doing laundry, cleaning a desk, or completing a small administrative task. These tasks may not be objectively difficult, but they can still feel psychologically heavy.  This project calls that hidden resistance **Invisible Friction**.  The central technical objective is not only to build a prediction model, but to study how different **sampling techniques** affect model error and inference quality.  ---  ## Project Identity  **Course:** 24AM4PCIST · Inferential Statistics   **Project Type:** Alternative Assessment Tool   **Domain:** Inferential Statistics, Behavioral Machine Learning, Sampling Design, Model Evaluation   **Primary Deliverable:** Sampling-aware ML pipeline + Streamlit demo + Report + PPT    ---  ## Core Idea  Most productivity systems measure what happens after a task begins:  - Was the task completed? - How long did it take? - Was the deadline met? - How productive was the user?  This project focuses on what happens **before** a task begins:  - How mentally heavy does the task feel? - Is the user likely to delay it? - Which factors increase resistance? - Does sampling design affect how well a model learns this behavior?  The project separates **actual difficulty** from **perceived initiation resistance**.  | Task | Actual Difficulty | Possible Friction | Reason | |---|---:|---:|---| | Replying to an email | Low | High | Social pressure or uncertainty | | Paying a bill | Low | Medium/High | Consequence severity and avoidance | | Walking the dog | Medium | Low | Clear reward and familiar routine | | Starting an assignment | Medium | High | Deadline pressure and unclear start point |  ---  ## Problem Statement  How can a machine learning model predict the invisible friction level of everyday tasks, and how do different sampling techniques affect model error and behavioral inference quality?  ---  ## Research Questions  ### Primary Research Question  How do different sampling techniques influence the predictive performance of machine learning models attempting to estimate invisible friction in everyday tasks?  ### Secondary Research Question  Which behavioral, contextual, and environmental factors contribute most strongly to high-friction task states?  ---  ## Hypothesis  The main hypothesis is that **Proportional Stratified Sampling** will produce lower model error than Simple Random Sampling, Balanced Stratified Sampling, and Cluster Sampling because invisible friction is heterogeneously distributed across task categories and friction levels.  Preserving subgroup representation should improve generalization.  ---  ## Key Result  The best-performing configuration was:  | Metric | Value | |---|---:| | Sampling Method | Proportional Stratified Sampling | | Model | Logistic Regression | | Accuracy | 0.7976 | | F1-score | 0.7966 | | Model Error | 0.2024 |  The result supports the central inferential statistics argument:  > Sampling design can affect model performance as much as model complexity.  ---  ## Why This Is an Inferential Statistics Project  This project treats the machine learning model as an experimental instrument.  The main study is not simply:  > Which model gives the highest accuracy?  The real study is:  > Which sampling technique produces the most reliable inference about the population?  In this project:  | Inferential Statistics Concept | Project Equivalent | |---|---| | Population | Full synthetic dataset of everyday task instances | | Sample | Training subset created using a sampling technique | | Estimator | Machine learning model trained on sampled data | | Sampling Error | Performance loss due to sample not representing the population | | Inference Quality | Model generalization on unseen task instances |  ---  ## Dataset  The dataset is synthetically generated because no public dataset directly measures invisible friction.  The generated dataset contains:  - **20,000 task instances** - **Task-level features** - **Psychological features** - **Environmental features** - **Numerical friction score** - **Categorical friction level**  Each row represents one task in a specific behavioral context.  ---  ## Dataset Features  | Feature | Type | Description | |---|---|---| | `task_type` | Categorical | Academic, household, communication, finance, health, personal administration | | `estimated_duration` | Numerical | Expected duration in minutes | | `deadline_pressure` | Categorical | Low, medium, or high urgency | | `time_of_day` | Categorical | Morning, afternoon, evening, or night | | `energy_level` | Numerical | User energy level from 1 to 10 | | `mood_level` | Numerical | User mood level from 1 to 10 | | `task_familiarity` | Categorical | New or repeated task | | `reward_clarity` | Categorical | Unclear, moderate, or clear reward | | `consequence_severity` | Numerical | Severity of delaying the task | | `previous_delay_count` | Numerical | Number of previous postponements | | `environment_noise` | Categorical | Low, medium, or high noise | | `device_distraction` | Categorical | Low, medium, or high digital distraction | | `social_obligation` | Categorical | Individual, family, college, or work-related | | `friction_score` | Numerical | Generated friction score | | `friction_level` | Target | Low, medium, or high friction |  ---  ## Friction Score Construction  The friction score is generated using a weighted additive behavioral model:  text
F = Wt + Wd + Wr + Wn + Wdis + Wf + Wtime
    + 0.12D + 3P + 1.5C
    - 2.2E - 1.8M + ε
 Where:  | Symbol | Meaning | |---|---| | `F` | Friction score | | `D` | Estimated task duration | | `P` | Previous delay count | | `C` | Consequence severity | | `E` | Energy level | | `M` | Mood level | | `ε` | Random noise |  The generated friction score is converted into three classes:  | Friction Score Range | Class | |---|---| | `F ≤ 35` | Low | | `35 < F ≤ 65` | Medium | | `F > 65` | High |  ---  ## Sampling Techniques Compared  Four sampling techniques were implemented and compared.  ### 1. Simple Random Sampling  Every row has an equal probability of selection.  **Strength:** Simple and unbiased in expectation.   **Weakness:** May underrepresent rare high-friction patterns by chance.  ---  ### 2. Proportional Stratified Sampling  Samples are drawn from strata in proportion to the population structure.  In this project, stratification is based on:  - `friction_level` - `task_type`  **Strength:** Preserves subgroup representation without distorting the population.   **Observed result:** Best-performing sampling technique.  ---  ### 3. Balanced Stratified Sampling  Equal samples are drawn from each friction class.  **Strength:** Increases minority class visibility.   **Weakness:** Distorts real population proportions.  This underperformed because the model was trained on an artificially balanced world that did not match the test distribution.  ---  ### 4. Cluster Sampling  Entire task-type clusters are selected.  **Strength:** Useful when natural groups exist.   **Weakness:** Can reduce behavioral diversity if selected clusters do not represent the full population.  Cluster sampling underperformed because selected task categories could not capture all behavioral contexts.  ---  ## Machine Learning Models Compared  The following models were trained on each sampling method:  1. Logistic Regression 2. Decision Tree Classifier 3. Random Forest Classifier 4. Gradient Boosting Classifier  This produced **16 model-sampling combinations**.  ---  ## Why Logistic Regression Won  Logistic Regression performed best because the generated friction score was mostly additive and linear.  This does not mean Logistic Regression is always superior. It means the model structure aligned well with the data-generating process.  This is an important conclusion:  > Higher model complexity does not automatically reduce error.  Random Forest and Gradient Boosting are stronger when there are complex nonlinear interactions. In this project, the underlying behavioral equation was largely linear, so Logistic Regression captured it efficiently.  ---  ## Best Results Summary  | Rank | Sampling Method | Model | Accuracy | F1-score | Model Error | |---:|---|---|---:|---:|---:| | 1 | Proportional Stratified | Logistic Regression | 0.7976 | 0.7966 | 0.2024 | | 2 | Simple Random | Logistic Regression | 0.7970 | 0.7960 | 0.2030 | | 3 | Cluster | Logistic Regression | 0.7858 | 0.7848 | 0.2142 | | 4 | Balanced Stratified | Logistic Regression | 0.7746 | 0.7693 | 0.2254 |  ---  ## Confusion Matrix of Best Model  | Actual / Predicted | Predicted Low | Predicted Medium | Predicted High | |---|---:|---:|---:| | Actual Low | 1540 | 392 | 0 | | Actual Medium | 327 | 2117 | 103 | | Actual High | 0 | 190 | 331 |  ### Key Interpretation  The model never confused actual low-friction tasks with high-friction tasks, and never confused actual high-friction tasks with low-friction tasks.  Most mistakes occurred between neighboring classes:  - Low ↔ Medium - Medium ↔ High  This is expected because invisible friction is a gradual psychological scale rather than a sharply separated category.  ---  ## Model Explainability  Since Logistic Regression was the best model, coefficient analysis was used to interpret its behavior.  ### High-Friction Predictors  The model identified the following as strong contributors to high friction:  - High deadline pressure - Unclear reward - Previous delay count - Night-time task timing - High device distraction - High environment noise - Low energy - Low mood - High consequence severity  ### Interpretation  High friction is not simply caused by task duration or objective difficulty.  It emerges from:  - urgency - ambiguity - accumulated delay - poor internal energy - low mood - environmental distraction - perceived consequences  ---  ## Project Pipeline text
Dataset Generation
        ↓
Descriptive Statistics
        ↓
Sampling Techniques
        ↓
Model Training
        ↓
Evaluation
        ↓
Explainability Analysis
        ↓
CLI + Streamlit Prediction Tool
 ---  ## Repository Structure text
invisible-friction-prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── invisible_friction_dataset.csv
│
├── outputs/
│   ├── best_model.joblib
│   ├── model_results.csv
│   ├── confusion_matrices/
│   ├── explainability/
│   ├── plots/
│   └── statistical_analysis/
│
├── reports/
│   └── project_notes.md
│
└── src/
    ├── generate_dataset.py
    ├── sampling.py
    ├── train_models.py
    ├── visualize.py
    ├── statistical_analysis.py
    ├── correlation_analysis.py
    ├── sampling_visuals.py
    ├── explain_best_model.py
    ├── save_best_model.py
    └── friction_predictor.py
 ---  ## Installation and Setup  ### 1. Clone the repository bash
git clone https://github.com/VrishinDutt/invisible-friction-prediction.git
cd invisible-friction-prediction
 ### 2. Create a virtual environment  For macOS/Linux: bash
python3 -m venv venv
source venv/bin/activate
 For Windows: bash
python -m venv venv
venv\Scripts\activate
 ### 3. Install dependencies bash
pip install -r requirements.txt
 ---  ## How to Run the Full Project  ### Step 1: Generate the dataset bash
python src/generate_dataset.py
 This creates: text
data/invisible_friction_dataset.csv
 ---  ### Step 2: Train models across sampling techniques bash
python src/train_models.py
 This creates: text
outputs/model_results.csv
outputs/confusion_matrices/
 ---  ### Step 3: Generate performance plots bash
python src/visualize.py
 This creates: text
outputs/plots/
 ---  ### Step 4: Generate statistical analysis bash
python src/statistical_analysis.py
python src/correlation_analysis.py
python src/sampling_visuals.py
 This creates: text
outputs/statistical_analysis/
 ---  ### Step 5: Generate explainability outputs bash
python src/explain_best_model.py
 This creates: text
outputs/explainability/
 ---  ### Step 6: Save the best model bash
python src/save_best_model.py
 This creates: text
outputs/best_model.joblib
 ---  ### Step 7: Run the command-line predictor bash
python src/friction_predictor.py
 ---  ### Step 8: Run the Streamlit app bash
streamlit run app.py
 Then open: text
http://localhost:8501
 ---  ## Streamlit Application  The Streamlit app allows the user to enter task conditions and receive:  - Predicted friction level - Prediction probabilities - Friction index from 0 to 100 - Likely friction contributors - Suggested intervention  ### Example Input  | Field | Value | |---|---| | Task Type | academic | | Estimated Duration | 90 | | Deadline Pressure | high | | Time of Day | night | | Energy Level | 3 | | Mood Level | 4 | | Task Familiarity | new | | Reward Clarity | unclear | | Consequence Severity | 8 | | Previous Delay Count | 5 | | Environment Noise | medium | | Device Distraction | high | | Social Obligation | college |  ### Expected Output text
Predicted Friction: HIGH

Likely Contributors:
- Low energy
- Repeated postponement
- High deadline pressure
- Unclear reward
- High device distraction

Suggested Intervention:
Break the task into a 5-minute starting step, reduce distractions, and clarify the immediate reward.
 ---  ## Friction Index  The app converts model probabilities into an intuitive 0–100 score. python
friction_index = (
    prob["low"] * 20 +
    prob["medium"] * 55 +
    prob["high"] * 90
)
 Interpretation:  | Friction Index | Meaning | |---:|---| | 0–35 | Low friction | | 35–65 | Medium friction | | 65–100 | High friction |  ---  ## Important Code Snippets  ### Sampling Function python
def stratified_sample(df, sample_size=5000):
    fraction = sample_size / len(df)

    sample = (
        df.groupby(["friction_level", "task_type"], group_keys=False)
          .sample(frac=fraction, random_state=42)
          .reset_index(drop=True)
    )

    return sample
 ### Training Pipeline python
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)
 ### Evaluation Metrics python
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
model_error = 1 - accuracy
 ---  ## Generated Outputs  After running the full project, the following outputs are produced:  | Output | Description | |---|---| | `model_results.csv` | Model performance across all sampling methods | | `confusion_matrices/` | Confusion matrices for each model-sampling pair | | `plots/` | F1-score and error comparison charts | | `statistical_analysis/` | Descriptive statistics, correlation matrix, sampling visuals | | `explainability/` | Logistic Regression coefficient analysis | | `best_model.joblib` | Saved best-performing model |  ---  ## Main Findings  ### 1. Sampling technique affected model error  The same model produced different results depending on the training sample.  ### 2. Proportional Stratified Sampling performed best  It preserved natural population proportions while ensuring subgroup representation.  ### 3. Balanced Stratified Sampling overcorrected  It forced equal class counts and introduced distribution mismatch.  ### 4. Cluster Sampling lost behavioral diversity  Selecting only certain task categories reduced generalization.  ### 5. Logistic Regression outperformed more complex models  The dataset was generated using mostly additive relationships, making Logistic Regression well suited.  ---  ## Limitations  1. The dataset is synthetic. 2. Behavioral weights are based on assumptions, not real survey data. 3. Human friction is subjective and may vary between individuals. 4. The model does not currently personalize predictions. 5. The current generation process is mostly linear. 6. Real-world validation is still required.  ---  ## Future Scope  Possible extensions:  - Collect real user task-friction data - Build personalized friction profiles - Add SHAP-based feature explanations - Add nonlinear interaction effects - Build a mobile application - Integrate with calendars and task managers - Convert into a productivity assistant - Compare more sampling strategies - Add cross-validation and confidence intervals - Use real survey-based labels  ---  ## Academic Conclusion  Invisible Friction Analytics demonstrates that sampling design plays a crucial role in behavioral machine learning.  The project shows that the best model is not necessarily the most complex model. Instead, performance depends on:  - how the dataset is generated - how the training sample is selected - how well the model assumptions match the data - how representative the sample is of the population  The strongest statistical conclusion is:  > A machine learning model is only as reliable as the sample used to train it.  ---  ## Team  | Name | USN | |---|---| | Tejas N A | 1BM24AI177 | | VC Mohit Rao | 1BM24AI82 | | Yashwanth J | 1BM24AI196 | | Vrishin Dutt CG | 1BM24AI195 |  ---  ## Course Details  | Field | Details | |---|---| | Course | 24AM4PCIST · Inferential Statistics | | Assessment | Alternative Assessment Tool | | Department | Artificial Intelligence and Machine Learning | | Institution | B.M.S. College of Engineering | | Faculty In-Charge | Dr Maithri K |  ---  ## Repository Status  This repository contains:  - Complete dataset generation pipeline - Four sampling techniques - Four machine learning models - Evaluation pipeline - Statistical analysis outputs - Explainability analysis - CLI predictor - Streamlit web app - Report and PPT-ready project narrative  ---  ## Final Note  This project is not just a friction classifier.  It is a sampling-aware behavioral inference study.  The model predicts invisible friction, but the project’s real contribution is showing how sampling technique changes what the model learns about human behavior.
:::
