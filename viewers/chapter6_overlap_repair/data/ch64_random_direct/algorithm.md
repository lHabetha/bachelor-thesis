# Random directions + direct verify (control)

Chapter 6.4 overlap-only strict repair (`random_direct__none`) on the frozen 50-start benchmark `strict_overlap_repair_ch64_v1`.

- Strict success (analytic overlap <= 1e-6): **40/50**
- Mean overlap-volume reduction (incl. failures): **95.8%**
- Mean verifier (analytic overlap) calls: **234**

Each frame shows the clevis at one accepted repair step; the overlap panel reports the analytic overlap volume, the strict `overlap_ok` flag, and the per-pair residuals. Geometry is generated on demand from the frame parameters.
