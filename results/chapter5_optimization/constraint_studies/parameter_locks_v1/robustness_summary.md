# Parameter Lock Robustness Study

Study ID: `parameter_locks_v1`

This suite reruns Chapter 5 optimizers while selected parameters are forced to remain
equal to their blocked-start values. Each constrained run is paired with its
unconstrained baseline.

## Run Summary

| Run | Constraint | Oracle OK | False OK | Mean Dist | Drop vs Baseline | Recoverability |
|-----|------------|-----------|----------|-----------|------------------|----------------|

## Parameter Criticality

| Parameter | Locked Starts | Mean Success Drop | Mean Distance Delta | False OK |
|-----------|---------------|-------------------|---------------------|----------|
