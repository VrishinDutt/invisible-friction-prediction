# Invisible Friction Prediction

## Core Idea

Invisible friction refers to the mental resistance experienced before starting an everyday task. Some tasks are objectively simple but feel difficult because of low energy, unclear rewards, deadline pressure, previous delay, distractions, or mood.

## Objective

To predict whether a task has low, medium, or high invisible friction using machine learning, and to compare how different sampling techniques affect model error.

## Sampling Techniques Used

1. Simple Random Sampling
2. Proportional Stratified Sampling
3. Balanced Stratified Sampling
4. Cluster Sampling

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier

## Best Result

Sampling Method: Proportional Stratified Sampling  
Model: Logistic Regression  
Accuracy: 0.7976  
F1-score: 0.7966  
Model Error: 0.2024  

## Interpretation

Proportional stratified sampling performed best because it preserved the real distribution of low, medium, and high friction cases while still ensuring representation across task types. This helped the model generalize better to unseen data.

Logistic regression performed better than more complex models because the synthetic behavioral rules used to generate friction scores were largely additive and linear. This shows that a more complex model does not automatically guarantee lower error.

## Confusion Matrix Interpretation

The model predicted low and medium friction cases well. Most errors occurred between neighboring classes, especially low vs medium and medium vs high. This is expected because friction exists on a gradual psychological scale rather than as sharply separated categories.

The model did not confuse low friction directly with high friction, which shows that it learned the broad structure of the problem correctly.

## Main Conclusion

The project demonstrates that sampling technique can influence model error as much as, or more than, model complexity. In this experiment, proportional stratified sampling produced the lowest error by preserving subgroup representation without distorting the natural population structure.
