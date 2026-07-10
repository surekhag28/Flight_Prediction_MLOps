"""
DAG: Retrain all models

Three triggers (all runs the same full pipeline)
    - Monthly               -- cron "0 2 * *" (1st of each month at 2:00 pm UTC)
    - Drift                 -- triggered by monitor_dag when drift detected
    - Performance drop      -- triggered by monitor_dag when model

Full pipeline run per execution:
    set_pipeline_run_id         -- creates HPO + pipeline parent (delay + congestion + anomaly) run
        - create_proxy_labels
        - [hpo_delay, hpo_congestion, hpo_anomaly] (parallel run for 30 trials each)
        - [train_delay, train_congestion, train_anomaly] (parallel run for full dataset)
        - evaluate_all_models
        - register_all_models
        - finalise_runs
"""
