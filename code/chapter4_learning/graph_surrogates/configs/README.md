# Configs

`protocol_v1.json` is the frozen graph-surrogate protocol. It records the input
data paths, architecture-screen grid, full fixed-label grid, training defaults,
and the conditional gate for any GNN-driven active-learning rerun.

`data.task22_trajectories` points at the frozen MLP summary folder
(`results/chapter4_label_efficiency/mlp_dense50k_labelblind/`).

Treat this file as frozen: if a protocol change is needed for a rerun, add a
new config file instead of editing this one, so existing results stay
reproducible.
