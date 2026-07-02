# MLP-only (multitask, binary-head stop)

Chapter 6.4 overlap-only strict repair (`mlp__multitask__taubin0.1`) on the frozen 50-start benchmark `strict_overlap_repair_ch64_v1`.

- Strict success (analytic overlap <= 1e-6): **11/50**
- Mean overlap-volume reduction (incl. failures): **57.4%**
- Mean verifier (analytic overlap) calls: **7**

Each frame shows the clevis at one accepted repair step; the overlap panel reports the analytic overlap volume, the strict `overlap_ok` flag, and the per-pair residuals. Geometry is generated on demand from the frame parameters.
