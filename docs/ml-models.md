# ML Models

## Overview

The platform trains and serves three ML models, each solving different aviation prediction task. All models are tracked in MLflow, versioned in MLflow model registry, and include structured model card for tracking and governance.



## Product A - Flight Delay Risk


### Task
 
Binary classification: predict the probability that a given flight will be delayed.

### Model

Best model is identified through hyperparameter tuning and used further for training the final classifier.

### Features

| Feature | Source | Description |
|------|-----|------|
| speed_ms | Request/Feast | Ground speed in m/s |
| altitude_m | Request/Feast | Barometric altitude in meters |
| vertical_rate_ms | Request/Feast | Climb/descent rate in m/s |
| heading_change_5m | Request/Feast | Heading change over 5 minutes (degrees) |
| avg_speed_15m | Request/Feast | Average speed over 15 minutes in m/s |
| aircraft_count_50km | Feast(congestion) | Aircraft density within 50 km radius |
| congestion_score | Request/Feast | Airport congestion (Congestion data) |
| hour_of_day | Request/Computed | UTC hour of day |
| day_of_week | Request/Computed | Day of week (0=Monday) |


### Target

Binary Label: 1= delayed, 0=not delayed. Labels are proxy-derived from trajectory patterns (see data-pipeline.md)

### Training

Source: `src/ml/train_delay.py`

```
CatBoost binary classifier
├── Data: gold/flights/ (up to HPO_SAMPLE_ROWS rows)
├── Split: 80/20 train/test, stratified by label
├── HPO: Optuna, 30 trials
│   ├── num_leaves: 16–256
│   ├── learning_rate: 1e-3–0.3 (log scale)
│   ├── n_estimators: 100–1000
│   ├── max_depth: 3–12
│   ├── min_child_samples: 5–100
│   └── subsample: 0.5–1.0
├── Eval metric: AUC-ROC
└── Quality gate: AUC-ROC > 0.65 (vs. current Production model)
```

### Evaluation

Source: `src/ml/evaluate.py`

`evaluate_delay()` loads the challenger and the current production champion model. If the challenger's AUC_ROC on the test set beats the champion by any margin, the challenger model will be promoted to Staging. If no production models exists then challenger will be promoted unconditionally.

### Output


| Field | Type | Description |
|------|-----|------|
| delay_probability | float 0-1 | Probability of delay |
| risk_label | LOW / MEDIUM / HIGH | Threshold-based label |
| top_features | list[str] | Top-3 features by model importance |

--------------------
