# Invisible Friction Analytics

## An Inferential Study of Sampling Techniques in Behavioral Machine Learning

Invisible Friction Analytics is an Inferential Statistics and Machine Learning project that predicts the hidden mental resistance people experience before starting everyday tasks.

The project investigates a subtle everyday phenomenon:

> Why do simple tasks sometimes feel disproportionately difficult to begin?

Examples include replying to a message, paying a bill, starting an assignment, doing laundry, cleaning a desk, or completing a small administrative task. These tasks may not be objectively difficult, but they can still feel psychologically heavy.

This project calls that hidden resistance **Invisible Friction**.

The central technical objective is not only to build a prediction model, but to study how different **sampling techniques** affect model error and inference quality.

---

## Project Identity

| Field | Details |
|---|---|
| Course | 24AM4PCIST · Inferential Statistics |
| Assessment | Alternative Assessment Tool |
| Department | Artificial Intelligence and Machine Learning |
| Institution | B.M.S. College of Engineering |
| Faculty In-Charge | Dr Maithri K |
| Domain | Inferential Statistics, Behavioral Machine Learning, Sampling Design, Model Evaluation |

---

## Team

| Name | USN |
|---|---|
| Tejas N A | 1BM24AI177 |
| VC Mohit Rao | 1BM24AI82 |
| Yashwanth J | 1BM24AI196 |
| Vrishin Dutt CG | 1BM24AI195 |

---

## Core Idea

Most productivity systems measure what happens after a task begins:

- Was the task completed?
- How long did it take?
- Was the deadline met?
- How productive was the user?

This project focuses on what happens **before** a task begins:

- How mentally heavy does the task feel?
- Is the user likely to delay it?
- Which factors increase resistance?
- Does sampling design affect how well a model learns this behavior?

The project separates **actual difficulty** from **perceived initiation resistance**.

| Task | Actual Difficulty | Possible Friction | Reason |
|---|---:|---:|---|
| Replying to an email | Low | High | Social pressure or uncertainty |
| Paying a bill | Low | Medium/High | Consequence severity and avoidance |
| Walking the dog | Medium | Low | Clear reward and familiar routine |
| Starting an assignment | Medium | High | Deadline pressure and unclear start point |

---

## Problem Statement

How can a machine learning model predict the invisible friction level of everyday tasks, and how do different sampling techniques affect model error and behavioral inference quality?

---

## Research Questions

### Primary Research Question

How do different sampling techniques influence the predictive performance of machine learning models attempting to estimate invisible friction in everyday tasks?

### Secondary Research Question

Which behavioral, contextual, and environmental factors contribute most strongly to high-friction task states?

---

## Hypothesis

The main hypothesis is that **Proportional Stratified Sampling** will produce lower model error than Simple Random Sampling, Balanced Stratified Sampling, and Cluster Sampling because invisible friction is heterogeneously distributed across task categories and friction levels.

Preserving subgroup representation should improve generalization.

---

## Why This Is an Inferential Statistics Project

This project treats the machine learning model as an experimental instrument.

The main study is not simply:

> Which model gives the highest accuracy?

The real study is:

> Which sampling technique produces the most reliable inference about the population?

In this project:

| Inferential Statistics Concept | Project Equivalent |
|---|---|
| Population | Full synthetic dataset of everyday task instances |
| Sample | Training subset created using a sampling technique |
| Estimator | Machine learning model trained on sampled data |
| Sampling Error | Performance loss due to sample not representing the population |
| Inference Quality | Model generalization on unseen task instances |

The strongest statistical conclusion is:

> A machine learning model is only as reliable as the sample used to train it.

---

## Dataset

The dataset is synthetically generated because no public dataset directly measures invisible friction.

The generated dataset contains:

- **20,000 task instances**
- **Task-level features**
- **Psychological features**
- **Environmental features**
- **Numerical friction score**
- **Categorical friction level**

Each row represents one task in a specific behavioral context.

---

## Dataset Features

| Feature | Type | Description |
|---|---|---|
| `task_type` | Categorical | Academic, household, communication, finance, health, personal administration |
| `estimated_duration` | Numerical | Expected duration in minutes |
| `deadline_pressure` | Categorical | Low, medium, or high urgency |
| `time_of_day` | Categorical | Morning, afternoon, evening, or night |
| `energy_level` | Numerical | User energy level from 1 to 10 |
| `mood_level` | Numerical | User mood level from 1 to 10 |
| `task_familiarity` | Categorical | New or repeated task |
| `reward_clarity` | Categorical | Unclear, moderate, or clear reward |
| `consequence_severity` | Numerical | Severity of delaying the task |
| `previous_delay_count` | Numerical | Number of previous postponements |
| `environment_noise` | Categorical | Low, medium, or high noise |
| `device_distraction` | Categorical | Low, medium, or high digital distraction |
| `social_obligation` | Categorical | Individual, family, college, or work-related |
| `friction_score` | Numerical | Generated friction score |
| `friction_level` | Target | Low, medium, or high friction |

---

## Invisible Friction Framework

The project models friction as a combination of three components:

```text
Invisible Friction = f(Task Properties, Psychological Context, Environmental Context)
